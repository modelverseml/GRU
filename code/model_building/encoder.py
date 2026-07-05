"""
Encoders + training data, all in one place.

This module owns everything between raw text and the arrays the model trains on:

  * builds each static encoder (word2vec / fasttext / glove) by downloading the
    pretrained vectors and TRIMMING them to our dataset's vocabulary in memory,
    so the full multi-gigabyte files are NEVER written to disk -- only the small
    trimmed copies under data/text_encoder_small/ are saved.
  * loads BERT straight from HuggingFace (bert-base-uncased) at runtime, so its
    weights aren't stored in the repo either.
  * turns text into a fixed-size (max_words, dim) sequence of vectors.
  * encodes the train/dev/test splits and saves them as .npz under
    data/text_embedding/ (so we don't re-encode on every training run).

Run it to (re)build everything the model needs:

    python encoder.py                 # build/trim encoders, then save embeddings
    python encoder.py --force         # re-download + re-trim the static encoders
    python encoder.py --encoder bert  # just one encoder

The training script and the serving backend both import from here
(sentences_encoder, load_embeddings, the encoder names, LABEL_TO_ID).
"""

import argparse
import shutil
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gensim.downloader as api
from gensim.models import KeyedVectors
# transformers is imported lazily inside the bert loader, so deployments that
# skip bert don't need it installed and never pay its import / download cost.

# ----------------------------- paths / names -----------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw"
# only the trimmed encoders are ever stored (the full ones are never saved)
ENCODER_ROOT = REPO_ROOT / "data" / "text_encoder_small"
# encoded splits get saved here so we don't re-encode every training run
EMBED_ROOT = REPO_ROOT / "data" / "text_embedding"

WORD2VEC = "word2vec"
FASTTEXT = "fasttext"
GLOVE = "glove"
BERT = "bert"

# gensim downloader ids for the three static encoders
GENSIM_IDS = {
    WORD2VEC: "word2vec-google-news-300",
    FASTTEXT: "fasttext-wiki-news-subwords-300",
    GLOVE: "glove-wiki-gigaword-300",
}
GENSIM_NAMES = [WORD2VEC, FASTTEXT, GLOVE]
# bert is pulled from HuggingFace by name (cached after the first download)
BERT_HF_NAME = "bert-base-uncased"

ALL_ENCODERS = [WORD2VEC, FASTTEXT, GLOVE, BERT]

# every title gets padded/cut to this many words
MAX_WORDS = 30

# the three splits made in data_generation.py
SPLITS = ["train", "dev", "test"]

# sentiment text -> class id (alphabetical order matches the data files)
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


# ----------------------- building: download + trim -----------------------

def dataset_vocab():
    # collect every word variant our encoding path could look up (strip
    # punctuation, try the word as-is and lowercased) so the trimmed encoder
    # keeps exactly what inference needs and nothing more.
    words = set()
    for jsonl in sorted(RAW_ROOT.glob("*.jsonl")):
        df = pd.read_json(jsonl, lines=True)
        for title in df["title"].astype(str):
            for tok in title.split():
                w = tok.strip(string.punctuation)
                if not w:
                    continue
                words.add(w)
                words.add(w.lower())
    return words


def build_trimmed_encoder(name, vocab, force=False):
    # download the full pretrained vectors, keep only our dataset's words, and
    # save just the small copy. the full vectors are trimmed in memory and never
    # written into data/; gensim's download cache is deleted afterwards.
    out_dir = ENCODER_ROOT / name
    out_file = out_dir / "vectors.kv"
    if out_file.exists() and not force:
        print(f"[{name}] trimmed encoder already exists, skipping")
        return

    gensim_id = GENSIM_IDS[name]
    print(f"[{name}] downloading {gensim_id} (full, in memory) ...")
    full = api.load(gensim_id)  # KeyedVectors, downloaded to the gensim cache

    keep = sorted(w for w in vocab if w in full.key_to_index)
    print(f"[{name}] keeping {len(keep)} / {len(vocab)} dataset words "
          f"(full vocab was {len(full)})")

    small = KeyedVectors(vector_size=full.vector_size)
    small.add_vectors(keep, np.stack([full[w] for w in keep]).astype(np.float32))

    out_dir.mkdir(parents=True, exist_ok=True)
    small.save(str(out_file))
    size_mb = sum(f.stat().st_size for f in out_dir.glob("vectors.kv*")) / 1e6
    print(f"[{name}] saved {len(small)} vectors -> {out_file}  (~{size_mb:.1f} MB)")

    # drop the full vectors + gensim's downloaded copy so nothing big is kept
    del full
    cache_dir = Path(api.BASE_DIR) / gensim_id
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"[{name}] removed download cache {cache_dir}")


# --------------------------- loading / encoding ---------------------------

# each encoder is loaded the first time it's used and cached, so we only pay for
# (and only need the deps for) the encoders we actually run. e.g. a deploy that
# never uses bert never imports transformers or downloads the bert weights.
_ENCODERS = {}


def _load_encoder(name):
    if name in GENSIM_NAMES:
        kv_file = ENCODER_ROOT / name / "vectors.kv"
        print(f"loading {name} from {kv_file}")
        return KeyedVectors.load(str(kv_file))
    if name == BERT:
        # imported here (not at module top) so skipping bert avoids needing it
        from transformers import BertTokenizer, BertModel
        print(f"loading bert from HuggingFace ({BERT_HF_NAME})")
        model = BertModel.from_pretrained(BERT_HF_NAME)
        model.eval()  # we only run it for embeddings, never train it here
        return {
            "tokenizer": BertTokenizer.from_pretrained(BERT_HF_NAME),
            "model": model,
        }
    raise ValueError(f"unknown encoder: {name}")


def get_encoder(name):
    # lazy-load + cache a single encoder by name
    if name not in _ENCODERS:
        _ENCODERS[name] = _load_encoder(name)
    return _ENCODERS[name]


def word_encoder(word, encoder_name):
    # look up one word in a static encoder. word2vec/glove return zeros for
    # words they don't know; we try the word as-is then lowercased.
    kv = get_encoder(encoder_name)
    word = word.strip(string.punctuation)
    for w in (word, word.lower()):
        try:
            return kv[w]
        except KeyError:
            continue
    return np.zeros(kv.vector_size, dtype=np.float32)


def sentences_encoder(sentence, encoder_name, max_words=MAX_WORDS):
    # turn a sentence into a fixed-size (max_words, dim) array of word vectors,
    # padding short sentences with zeros and cutting off long ones.
    if encoder_name == BERT:
        return _bert_encode(sentence, max_words)

    kv = get_encoder(encoder_name)
    dim = kv.vector_size

    vectors = []
    for word in sentence.split(' '):
        if not word.strip():
            continue  # skip empty bits from double spaces
        vectors.append(word_encoder(word, encoder_name))
        if len(vectors) == max_words:
            break  # sentence longer than max_words -> stop here

    while len(vectors) < max_words:
        vectors.append(np.zeros(dim, dtype=np.float32))  # pad to a fixed length

    return np.array(vectors, dtype=np.float32)


def _bert_encode(sentence, max_words):
    # bert is contextual: feed the whole sentence and take the per-token hidden
    # states. shape comes out (max_words, 768).
    bert = get_encoder(BERT)
    tokens = bert["tokenizer"](
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_words,
    )
    with torch.no_grad():
        out = bert["model"](**tokens)
    return out.last_hidden_state.squeeze(0).numpy()


# ----------------------- splits -> saved embeddings -----------------------

def label_to_id(sentiment):
    # split files may store the label as text ("negative") or as the number (0)
    if isinstance(sentiment, str):
        return LABEL_TO_ID[sentiment]
    return int(sentiment)


def encode_split(df, encoder_name, max_words=MAX_WORDS):
    # encode every title into (max_words, dim); X -> (n, max_words, dim), y -> (n,)
    X, y = [], []
    for title, sentiment in zip(df["title"], df["sentiment"]):
        X.append(sentences_encoder(str(title), encoder_name, max_words=max_words))
        y.append(label_to_id(sentiment))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_train_data(encoder_name, max_words=MAX_WORDS):
    # encode all three splits for one encoder, returns {split: (X, y)}
    data = {}
    for split in SPLITS:
        df = pd.read_json(RAW_ROOT / f"{split}.jsonl", lines=True)
        X, y = encode_split(df, encoder_name, max_words=max_words)
        print(f"{encoder_name}/{split}: X={X.shape} y={y.shape}")
        data[split] = (X, y)
    return data


def save_embeddings(encoder_name, max_words=MAX_WORDS):
    # encode the splits and save them so training doesn't re-encode every run
    data = generate_train_data(encoder_name, max_words=max_words)
    out_dir = EMBED_ROOT / encoder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, (X, y) in data.items():
        out_file = out_dir / f"{split}.npz"
        np.savez_compressed(out_file, X=X, y=y)
        print(f"saved {encoder_name}/{split}: X={X.shape} -> {out_file}")
    return data


def load_embeddings(encoder_name):
    # read the saved embeddings back. run encoder.py first to create them.
    in_dir = EMBED_ROOT / encoder_name
    data = {}
    for split in SPLITS:
        in_file = in_dir / f"{split}.npz"
        if not in_file.exists():
            raise FileNotFoundError(f"{in_file} not found -- run encoder.py first")
        npz = np.load(in_file)
        data[split] = (npz["X"], npz["y"])
        print(f"loaded {encoder_name}/{split}: X={npz['X'].shape}")
    return data


# --------------------------------- main ---------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="build/trim encoders and save the encoded splits"
    )
    p.add_argument("--encoder", choices=["all", WORD2VEC, FASTTEXT, GLOVE, BERT],
                   default="all", help="which encoder(s) to build + encode")
    p.add_argument("--max_words", type=int, default=MAX_WORDS,
                   help="words each title is padded/cut to")
    p.add_argument("--force", action="store_true",
                   help="re-download + re-trim the static encoders even if present")
    return p.parse_args()


def main():
    args = parse_args()
    encoders = ALL_ENCODERS if args.encoder == "all" else [args.encoder]

    # 1) build the trimmed static encoders among the requested set (bert is
    #    fetched from HuggingFace when first used, nothing to build).
    static_needed = [e for e in encoders if e in GENSIM_NAMES]
    if static_needed:
        print("building dataset vocabulary from", RAW_ROOT)
        vocab = dataset_vocab()
        print(f"dataset vocabulary: {len(vocab)} unique word forms")
        for name in static_needed:
            build_trimmed_encoder(name, vocab, force=args.force)
    if BERT in encoders:
        print("bert: loaded from HuggingFace at runtime (nothing to build)")

    # 2) encode the splits and save the embeddings for each requested encoder
    for encoder_name in encoders:
        save_embeddings(encoder_name, max_words=args.max_words)

    print("label mapping:", LABEL_TO_ID)


if __name__ == "__main__":
    main()
