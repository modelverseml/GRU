"""
Train a GRU to classify review sentiment (negative / neutral / positive).

We build the same model three ways -- PyTorch, TensorFlow, and a from-scratch
NumPy implementation (ManualGRU) -- train each on every encoder's data
(word2vec, fasttext, glove, bert), report the test accuracy, and save every
trained model into data/model_artifacts/ as its own file:

    pytorch_<enc>.pt      tensorflow_<enc>.keras      manual_<enc>.npz
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# NOTE: tensorflow is imported lazily (inside the TF functions) so this module
# can be imported without it -- the Streamlit deploy skips TF to save memory.

from encoder import load_embeddings, LABEL_TO_ID, WORD2VEC, FASTTEXT, GLOVE, BERT
from manual_gru import ManualGRU

# where the trained models get saved
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_ROOT = REPO_ROOT / 'data' / 'model_artifacts'

ENCODERS = [WORD2VEC, FASTTEXT, GLOVE]   # BERT skipped: 4.7 GB embeddings OOM locally
NUM_CLASSES = len(LABEL_TO_ID)  # 3: negative / neutral / positive

# training settings
HIDDEN_DIM = 128
NUM_LAYERS = 2      # how many GRU layers to stack (pytorch/tensorflow)
DROPOUT = 0.3       # applied between stacked layers and in the classifier head
EPOCHS = 5
BATCH_SIZE = 64
MAX_WORDS = 30

# the from-scratch numpy GRU is a single-layer cell, so a smaller hidden size
# keeps the pure-python BPTT fast while still training fine.
MANUAL_HIDDEN = 128

# fix the random state so runs are reproducible across learning-rate sweeps.
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:  # tensorflow is optional (the deploy may skip it)
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# learning rate per framework and per encoder; anything unlisted uses DEFAULT_LR.
DEFAULT_LR = 1e-3
LEARNING_RATES = {
    "pytorch": {WORD2VEC: 3e-3, FASTTEXT: 3e-3, GLOVE: 3e-3, BERT: 1e-3},
    "tensorflow": {WORD2VEC: 3e-3, FASTTEXT: 3e-3, GLOVE: 3e-3, BERT: 1e-3},
    "manual": {WORD2VEC: 5e-3, FASTTEXT: 5e-3, GLOVE: 5e-3, BERT: 3e-3},
}


def get_lr(framework, encoder_name):
    return LEARNING_RATES.get(framework, {}).get(encoder_name, DEFAULT_LR)


# ----------------------------- PyTorch -----------------------------

class GRUTorch(nn.Module):
    # GRU -> take the last real word -> small MLP head -> 3 classes
    def __init__(self, input_dim, hidden_dim, num_classes,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)       # out: (batch, seq_len, hidden)
        # titles are zero-padded at the end -> take the hidden state at the LAST
        # real word, not at timestep -1 (which would be a pad position).
        mask = (x.abs().sum(dim=-1) > 0)
        lengths = mask.sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        batch_idx = torch.arange(x.size(0), device=x.device)
        last = out[batch_idx, last_idx]
        return self.fc(last)


def evaluate_torch(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    return float((preds == y).mean())


def train_torch(data, encoder_name):
    X_train, y_train = data["train"]
    X_dev, y_dev = data["dev"]
    X_test, y_test = data["test"]

    input_dim = X_train.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed()
    model = GRUTorch(input_dim, HIDDEN_DIM, NUM_CLASSES).to(device)
    lr = get_lr("pytorch", encoder_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"  [torch] lr={lr}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_dev_acc = -1.0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        dev_acc = evaluate_torch(model, X_dev, y_dev, device)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [torch] epoch {epoch + 1}/{EPOCHS}  dev_acc={dev_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [torch] best dev_acc={best_dev_acc:.3f}")
    test_acc = evaluate_torch(model, X_test, y_test, device)

    out_file = ARTIFACT_ROOT / f"pytorch_{encoder_name}.pt"
    torch.save(model.state_dict(), out_file)
    print(f"  [torch] saved -> {out_file}")
    return test_acc


# ---------------------------- TensorFlow ----------------------------

def build_tf_model(input_dim, lr):
    import tensorflow as tf  # lazy: only needed when training the TF model
    model = tf.keras.Sequential([
        tf.keras.layers.Input((MAX_WORDS, input_dim)),
        tf.keras.layers.GRU(HIDDEN_DIM),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_tf(data, encoder_name):
    X_train, y_train = data["train"]
    X_dev, y_dev = data["dev"]
    X_test, y_test = data["test"]

    lr = get_lr("tensorflow", encoder_name)
    print(f"  [tf] lr={lr}")
    set_seed()
    model = build_tf_model(X_train.shape[-1], lr)
    model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,  # per-epoch progress bar (TF RNNs are slow on CPU)
    )
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    out_file = ARTIFACT_ROOT / f"tensorflow_{encoder_name}.keras"
    model.save(out_file)
    print(f"  [tf] saved -> {out_file}")
    return float(test_acc)


# ------------------------------ manual ------------------------------

# the from-scratch numpy GRU lives in its own module (manual_gru.py)
def train_manual(data, encoder_name):
    X_train, y_train = data["train"]
    X_dev, y_dev = data["dev"]
    X_test, y_test = data["test"]

    input_dim = X_train.shape[-1]
    lr = get_lr("manual", encoder_name)
    print(f"  [manual] lr={lr}")

    set_seed()
    model = ManualGRU(input_dim, MANUAL_HIDDEN, NUM_CLASSES)
    model.fit(X_train, y_train, X_dev, y_dev,
              epochs=EPOCHS, batch_size=BATCH_SIZE, lr=lr)
    test_acc = model.accuracy(X_test, y_test)

    out_file = ARTIFACT_ROOT / f"manual_{encoder_name}.npz"
    model.save(out_file)
    print(f"  [manual] saved -> {out_file}")
    return test_acc


# ------------------------------- main -------------------------------

def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    results = {}
    for encoder_name in ENCODERS:
        print(f"\n=== encoder: {encoder_name} ===")
        data = load_embeddings(encoder_name)
        results[encoder_name] = {
            "pytorch": train_torch(data, encoder_name),
            # "tensorflow": train_tf(data, encoder_name),  # skipped: TF RNNs are slow on CPU
            "tensorflow": None,
            "manual": train_manual(data, encoder_name),
        }

    print("\n=== test accuracy ===")
    print(f"{'encoder':<10} {'pytorch':>8} {'tensorflow':>11} {'manual':>8}")
    for encoder_name, accs in results.items():
        def show(v):
            return f"{v:.3f}" if v is not None else "-"
        print(f"{encoder_name:<10} {show(accs['pytorch']):>8} "
              f"{show(accs['tensorflow']):>11} {show(accs['manual']):>8}")


if __name__ == "__main__":
    main()
