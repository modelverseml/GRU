"""
Framework-agnostic prediction core for the sentiment app.

Given a piece of text, encode it with each text encoder (word2vec, fasttext,
glove, bert) and run all THREE trained models (the PyTorch GRU, the TensorFlow
SimpleRNN, and the from-scratch NumPy GRU) for that encoder, returning the
predicted label and the class confidences for every (encoder, framework) combo.

This module deliberately has NO web framework in it, so the FastAPI backend and
a future Streamlit app can both `import predictor` and call `predict(text)`.
"""

import os
import sys
from pathlib import Path

# quiet tensorflow's startup logging before it gets imported downstream
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# the training code lives in ../model_building -- put it on the path so we can
# reuse the exact same encoders, model class and hyper-parameters used to train.
MB_DIR = Path(__file__).resolve().parent.parent / "model_building"
sys.path.insert(0, str(MB_DIR))

import numpy as np
import torch
import torch.nn.functional as F
# tensorflow is imported lazily (inside _load_tf) so a deploy that skips the
# TF models doesn't need it installed and never pays its (large) memory cost.

# reuse training-time definitions so the architecture matches the saved weights
from encoder import (  # noqa: E402
    sentences_encoder, WORD2VEC, FASTTEXT, GLOVE, BERT, MAX_WORDS, LABEL_TO_ID,
)
from model_artifacts_generation import (  # noqa: E402
    GRUTorch, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_CLASSES, ARTIFACT_ROOT,
)
from manual_gru import ManualGRU  # noqa: E402

# alias so the rest of this file stays architecture-agnostic
TorchModel = GRUTorch
ManualModel = ManualGRU


def _env_list(name, default):
    # parse a comma-separated env var (e.g. VRNN_FRAMEWORKS="pytorch,manual")
    val = os.environ.get(name, "").strip()
    return [x.strip() for x in val.split(",") if x.strip()] if val else list(default)


# which encoders / models to run -- configurable so the Streamlit deploy can drop
# heavy pieces (e.g. VRNN_FRAMEWORKS="pytorch,manual" skips TensorFlow).
ENCODERS = _env_list("VRNN_ENCODERS", [WORD2VEC, FASTTEXT, GLOVE, BERT])
FRAMEWORKS = _env_list("VRNN_FRAMEWORKS", ["pytorch", "tensorflow", "manual"])
# id -> label ("negative" / "neutral" / "positive"), used to name the outputs
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
LABELS = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]

# loaded models get cached here so we only read each artifact from disk once
_torch_models = {}   # encoder -> TorchModel
_tf_models = {}      # encoder -> keras model
_manual_models = {}  # encoder -> ManualGRU


def _encode(text, encoder_name):
    # (max_words, dim) float32 array, then add a batch dim -> (1, max_words, dim)
    arr = sentences_encoder(str(text), encoder_name, max_words=MAX_WORDS)
    return np.asarray(arr, dtype=np.float32)[None, ...]


def _load_torch(encoder_name, input_dim):
    # build the same architecture used in training and load the saved weights
    if encoder_name not in _torch_models:
        path = ARTIFACT_ROOT / f"pytorch_{encoder_name}.pt"
        model = TorchModel(
            input_dim, HIDDEN_DIM, NUM_CLASSES,
            num_layers=NUM_LAYERS, dropout=DROPOUT,
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _torch_models[encoder_name] = model
    return _torch_models[encoder_name]


def _load_tf(encoder_name):
    if encoder_name not in _tf_models:
        import tensorflow as tf  # lazy: only imported if the TF models are used
        path = ARTIFACT_ROOT / f"tensorflow_{encoder_name}.keras"
        _tf_models[encoder_name] = tf.keras.models.load_model(path)
    return _tf_models[encoder_name]


def _load_manual(encoder_name):
    # the .npz holds the weight matrices; shapes are inferred from them
    if encoder_name not in _manual_models:
        path = ARTIFACT_ROOT / f"manual_{encoder_name}.npz"
        _manual_models[encoder_name] = ManualModel.load(path)
    return _manual_models[encoder_name]


def _format(encoder_name, framework, probs):
    # probs: 1d array of length NUM_CLASSES -> a tidy result row
    probs = [float(p) for p in probs]
    top = int(np.argmax(probs))
    return {
        "encoder": encoder_name,
        "framework": framework,
        "label": ID_TO_LABEL[top],
        "confidence": probs[top],
        "probabilities": {LABELS[i]: probs[i] for i in range(NUM_CLASSES)},
    }


def _predict_torch(encoder_name, x):
    model = _load_torch(encoder_name, input_dim=x.shape[-1])
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        probs = F.softmax(logits, dim=1).numpy()[0]
    return _format(encoder_name, "pytorch", probs)


def _predict_tf(encoder_name, x):
    model = _load_tf(encoder_name)
    # the keras model already ends in a softmax, so this is class probabilities
    probs = model.predict(x, verbose=0)[0]
    return _format(encoder_name, "tensorflow", probs)


def _predict_manual(encoder_name, x):
    model = _load_manual(encoder_name)
    probs = model.predict_proba(x)[0]   # from-scratch numpy RNN, returns softmax
    return _format(encoder_name, "manual", probs)


_PREDICTORS = {
    "pytorch": _predict_torch,
    "tensorflow": _predict_tf,
    "manual": _predict_manual,
}


def predict(text):
    # run every (encoder, framework) pair and return a flat list of result rows.
    # if a model's artifact hasn't been trained yet we skip it instead of failing
    # the whole request, so the app still works with whatever models are present.
    results = []
    for encoder_name in ENCODERS:
        x = _encode(text, encoder_name)
        for framework in FRAMEWORKS:
            try:
                results.append(_PREDICTORS[framework](encoder_name, x))
            except (FileNotFoundError, ValueError, OSError, ImportError) as e:
                # untrained artifact (keras raises ValueError, torch/np raise
                # FileNotFoundError) or a missing lib -> skip that model gracefully
                print(f"  [predict] skipping {framework}/{encoder_name}: {type(e).__name__}")
    return results


def warmup():
    # load the encoders + models up front so the first real request is fast, and
    # report only the (encoder, framework) that ACTUALLY produced a result -- i.e.
    # those whose artifacts exist. this is what /info exposes, so the UI describes
    # only what's really loaded (not the configured-but-untrained ones).
    rows = predict("good")
    encoders = [e for e in ENCODERS if any(r["encoder"] == e for r in rows)]
    frameworks = [f for f in FRAMEWORKS if any(r["framework"] == f for r in rows)]
    return {
        "encoders": encoders,
        "labels": LABELS,
        "frameworks": frameworks,
    }


if __name__ == "__main__":
    # quick manual check: python predictor.py "this product is amazing"
    sample = sys.argv[1] if len(sys.argv) > 1 else "this product is amazing"
    from pprint import pprint
    pprint(predict(sample))
