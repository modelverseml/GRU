"""Data preparation and inference helpers for the from-scratch GRU.

This module sits between raw text and the multi-layer `GRU` class in
`gru_scratch.py`:

* `generate_dataset` turns a corpus into the sliding-window training tensors the
  GRU expects, in either character mode (one-hot inputs) or word mode
  (pre-trained word-vector inputs).
* `train_test_split` carves those tensors into train and test partitions.
* `evaluate` scores a trained model on held-out data.
* `predict_next` / `generate` run a *trained* model forward to produce text.

These helpers only touch a model's `predict` method, so the same calls drive the
manual GRU and the TensorFlow / PyTorch wrappers identically.

Tensor convention (batch-first, shared with `gru_scratch.GRU`):
    m   : number of training sequences (batch size)
    T_x : time steps per sequence
    n_x : input feature size  (vocab size for chars, vector size for words)
    n_y : output feature size (vocab size)
so inputs are (m, T_x, n_x) and targets are (m, T_x, n_y).
"""

import numpy as np


def generate_dataset(words, T_x, word_vectors):
    """Build sliding-window training tensors from a corpus of words.

    A window of length ``T_x`` is slid over the corpus. For each window the input
    is the chunk of words and the target is that same chunk shifted one word to the
    left, so the model learns to predict the *next* word at every step. Inputs are
    dense word-embedding vectors; targets are one-hot over the vocabulary.

    Args:
        words        : the corpus as a list of words.
        T_x          : number of time steps per training sequence.
        word_vectors : a gensim KeyedVectors / dict mapping word -> vector.

    Returns:
        input_sequences  : embedding inputs, shape (m, T_x, n_x)
        output_sequences : one-hot targets,  shape (m, T_x, n_y)
        vocab_to_index   : dict mapping word -> feature index
        index_to_vocab   : dict mapping feature index -> word (used to decode)
    """
    # Drop any word the embedding doesn't know about -- there is no vector to feed
    # the network for an out-of-vocabulary word. gensim exposes membership via
    # `.key_to_index`; a plain dict is queried directly.
    try:
        words = [w for w in words if w in word_vectors.key_to_index]
    except AttributeError:
        words = [w for w in words if w in word_vectors]

    # Build the vocabulary: the sorted set of unique words, plus the two maps that
    # translate between a word and its column index in the one-hot targets.
    vocabs = sorted(list(set(words)))
    vocab_to_index = {c: i for i, c in enumerate(vocabs)}
    index_to_vocab = {i: c for i, c in enumerate(vocabs)}

    # Input feature size is the embedding dimension (gensim exposes it as
    # `.vector_size`; for a plain dict we read the length of any vector); output
    # feature size is the vocabulary size.
    try:
        n_x = word_vectors.vector_size
    except AttributeError:
        n_x = len(word_vectors[vocabs[0]])
    n_y = len(vocabs)
    m = len(words) - T_x - 1            # number of windows that fit

    input_sequences = np.zeros((m, T_x, n_x))
    output_sequences = np.zeros((m, T_x, n_y))

    # Fill window i, step t with word (i+t); the target is the next word (i+t+1).
    for i in range(m):
        for t in range(T_x):
            input_sequences[i, t, :] = word_vectors[words[i+t]]         # dense word vector
            output_sequences[i, t, vocab_to_index[words[i+t+1]]] = 1    # one-hot next word

    print(f"Generated {m} training sequences of length {T_x} from a corpus of "
          f"{len(words)} words, with a vocabulary of {len(vocabs)} unique words.")
    return input_sequences, output_sequences, vocab_to_index, index_to_vocab


def train_test_split(X, Y, test_size=0.2, shuffle=True, seed=0):
    """Split the generated tensors into train and test partitions.

    The sequences live along axis 0 (the ``m`` axis), so the split is taken
    over that axis and the same row indices are used for X and Y to keep
    each input paired with its target.

    Args:
        X         : input tensor,  shape (m, T_x, n_x).
        Y         : target tensor, shape (m, T_x, n_y).
        test_size : fraction of the ``m`` sequences to hold out for testing.
        shuffle   : shuffle the sequence order before splitting.
        seed      : RNG seed so the split is reproducible.

    Returns:
        X_train, X_test, Y_train, Y_test
    """
    m = X.shape[0]
    indices = np.arange(m)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)

    n_test = int(round(m * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    print(f"Split {m} sequences -> {X_train.shape[0]} train / {X_test.shape[0]} test.")
    return X_train, X_test, Y_train, Y_test


def evaluate(model, X, Y):
    """Score a trained model on data X/Y.

    For classification returns next-token accuracy (fraction of timesteps whose
    argmax prediction matches the one-hot target). For regression returns the
    mean-squared-error loss. Lower MSE / higher accuracy is better.
    """
    y_pred = model.predict(X)                       # (m, T_x, n_y)
    if model.task == "classification":
        # Compare predicted class (argmax over the feature axis) against the target.
        correct = np.argmax(y_pred, axis=-1) == np.argmax(Y, axis=-1)
        return float(np.mean(correct))
    return model.compute_loss(y_pred, Y)


def _encode_sequence(words, model, embedding):
    """Encode a list of words into one input sequence of shape (1, T, n_x).

    Each word becomes its embedding vector -- the same encoding `generate_dataset`
    produces, for a single sequence.
    """
    T = len(words)
    x = np.zeros((1, T, model.n_x))
    for t, word in enumerate(words):
        x[0, t, :] = embedding[word]               # dense word vector
    return x


def predict_next(model, embedding, decoder, seed_word):
    """Return the single most likely next word after ``seed_word`` (argmax).

    Works for any model exposing ``predict`` (the manual GRU and the TensorFlow
    / PyTorch wrappers all do), so the same call compares across frameworks.

    Args:
        model     : a trained model with a ``predict`` method and ``n_x``/``n_y``.
        embedding : word -> input-vector lookup (gensim KeyedVectors / dict).
        decoder   : ``index_to_vocab`` map, used to turn the predicted index back
                    into a word.
        seed_word : the word to condition on.
    """
    x = _encode_sequence([seed_word], model, embedding)
    y_pred = model.predict(x)                      # (1, T, n_y)
    idx = int(np.argmax(y_pred[0, -1, :]))         # distribution at the last step
    return decoder[idx]


def generate(model, embedding, decoder, seed_word, num_words=50, sample=True):
    """Generate text autoregressively, starting from ``seed_word``.

    At each step the words generated so far are re-encoded and run through
    ``model.predict``; the distribution at the final timestep gives the next word,
    which is appended and fed back in. Because it relies only on ``predict``, the
    same routine drives the manual GRU and the TensorFlow / PyTorch wrappers
    identically (no per-model hidden-state handling).

    Args:
        model     : a trained model with a ``predict`` method.
        embedding : word -> input-vector lookup (see ``predict_next``).
        decoder   : ``index_to_vocab`` map for turning indices back into words.
        seed_word : the word to start generation from.
        num_words : how many words to generate after the seed.
        sample    : True  -> draw the next word from the predicted distribution
                             (varied output);
                    False -> always take the argmax (deterministic, tends to repeat).
    """
    words = [seed_word]

    for _ in range(num_words):
        x = _encode_sequence(words, model, embedding)
        y_pred = model.predict(x)                  # (1, T, n_y)
        probs = y_pred[0, -1, :]                   # next-word scores at last step
        if sample:
            probs = np.clip(probs, 1e-12, None)
            probs = probs / probs.sum()            # renormalise to a valid pmf
            idx = int(np.random.choice(model.n_y, p=probs))
        else:
            idx = int(np.argmax(probs))
        words.append(decoder[idx])

    return ' '.join(words)