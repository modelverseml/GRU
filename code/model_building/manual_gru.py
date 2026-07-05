"""
A GRU written from scratch in NumPy -- no deep-learning library.

Same role as manual_rnn.py / manual_lstm.py, but a Gated Recurrent Unit: two
gates (update z, reset r) and a reset-gated candidate, with no separate cell
state. Single layer, take the hidden state at the last real word, then a linear
layer into the class scores. Trained with full backprop-through-time (BPTT) and
Adam, with gradient clipping and best-dev checkpointing.

Batch-first, row-vector convention; gates share a [a_prev, x] concatenation
(each gate weight is (hidden, hidden + input_dim)):

    concat = [a_prev, x]
    z  = sigmoid(concat   @ Wz.T + bz)           update gate
    r  = sigmoid(concat   @ Wr.T + br)           reset gate
    h~ = tanh   ([r*a_prev, x] @ Wh.T + bh)      candidate (reset-gated history)
    a  = (1 - z) * a_prev + z * h~               interpolate old vs candidate
"""

import numpy as np

DEFAULT_SEED = 42
_GATES = ("Wz", "bz", "Wr", "br", "Wh", "bh")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class ManualGRU:
    def __init__(self, input_dim=None, hidden_dim=None, num_classes=None,
                 seed=DEFAULT_SEED):
        if input_dim is None:
            return  # empty shell, to be filled by load()
        rng = np.random.default_rng(seed)
        concat = hidden_dim + input_dim
        scale = 1.0 / np.sqrt(concat)
        self.params = {
            "Wz": rng.standard_normal((hidden_dim, concat)) * scale,
            "Wr": rng.standard_normal((hidden_dim, concat)) * scale,
            "Wh": rng.standard_normal((hidden_dim, concat)) * scale,
            "bz": np.zeros((1, hidden_dim)),
            "br": np.zeros((1, hidden_dim)),
            "bh": np.zeros((1, hidden_dim)),
            "Wy": rng.standard_normal((num_classes, hidden_dim)) / np.sqrt(hidden_dim),
            "by": np.zeros((1, num_classes)),
        }
        self._init_adam()

    @property
    def hidden_dim(self):
        return self.params["Wz"].shape[0]

    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    @staticmethod
    def _last_real_idx(X):
        mask = np.abs(X).sum(axis=-1) > 0
        lengths = np.clip(mask.sum(axis=1), 1, None)
        return lengths - 1

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        B, T, _ = X.shape
        H = self.hidden_dim
        p = self.params

        a_prev = np.zeros((B, H))
        a_seq = np.zeros((B, T, H))
        cache = []
        for t in range(T):
            xt = X[:, t, :]
            concat = np.concatenate([a_prev, xt], axis=1)
            zt = _sigmoid(concat @ p["Wz"].T + p["bz"])
            rt = _sigmoid(concat @ p["Wr"].T + p["br"])
            concat_r = np.concatenate([rt * a_prev, xt], axis=1)
            hh = np.tanh(concat_r @ p["Wh"].T + p["bh"])
            a_next = (1 - zt) * a_prev + zt * hh
            a_seq[:, t, :] = a_next
            cache.append((a_prev, zt, rt, hh, xt))
            a_prev = a_next

        last_idx = self._last_real_idx(X)
        a_last = a_seq[np.arange(B), last_idx]
        logits = a_last @ p["Wy"].T + p["by"]
        return logits, (cache, last_idx, a_last, B, T, H)

    def backward(self, logits, y, ctx):
        cache, last_idx, a_last, B, T, H = ctx
        p = self.params

        probs = self._softmax(logits)
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B

        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["Wy"] = dlogits.T @ a_last
        grads["by"] = dlogits.sum(axis=0, keepdims=True)

        dA = np.zeros((B, T, H))
        dA[np.arange(B), last_idx] = dlogits @ p["Wy"]

        da_next = np.zeros((B, H))
        for t in reversed(range(T)):
            a_prev, zt, rt, hh, xt = cache[t]
            da = dA[:, t, :] + da_next

            # through a = (1-z)*a_prev + z*hh
            dzt = da * (hh - a_prev)
            dhh = da * zt
            da_prev = da * (1 - zt)                       # direct skip path

            # candidate hh = tanh([r*a_prev, x] @ Wh.T + bh)
            dhraw = dhh * (1 - hh ** 2)
            concat_r = np.concatenate([rt * a_prev, xt], axis=1)
            grads["Wh"] += dhraw.T @ concat_r
            grads["bh"] += dhraw.sum(axis=0, keepdims=True)
            dconcat_r = dhraw @ p["Wh"]
            d_rgated = dconcat_r[:, :H]                   # grad into (r * a_prev)
            drt = d_rgated * a_prev
            da_prev += d_rgated * rt

            concat = np.concatenate([a_prev, xt], axis=1)
            # update gate
            dzraw = dzt * zt * (1 - zt)
            grads["Wz"] += dzraw.T @ concat
            grads["bz"] += dzraw.sum(axis=0, keepdims=True)
            da_prev += (dzraw @ p["Wz"])[:, :H]
            # reset gate
            drraw = drt * rt * (1 - rt)
            grads["Wr"] += drraw.T @ concat
            grads["br"] += drraw.sum(axis=0, keepdims=True)
            da_prev += (drraw @ p["Wr"])[:, :H]

            da_next = da_prev
        return grads

    def _clip(self, grads, max_norm=5.0):
        total = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        if total > max_norm:
            for k in grads:
                grads[k] *= max_norm / (total + 1e-6)

    def _adam_step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k in self.params:
            self._m[k] = b1 * self._m[k] + (1 - b1) * grads[k]
            self._v[k] = b2 * self._v[k] + (1 - b2) * (grads[k] ** 2)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    # ---- public API (matches ManualRNN / ManualLSTM) ----
    def predict_proba(self, X):
        logits, _ = self.forward(X)
        return self._softmax(logits)

    def accuracy(self, X, y):
        return float((self.predict_proba(X).argmax(axis=1) == y).mean())

    def fit(self, X_train, y_train, X_dev, y_dev, epochs, batch_size, lr):
        self._init_adam()
        n = len(X_train)
        best_dev, best_params = -1.0, None
        for epoch in range(epochs):
            order = np.random.permutation(n)
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                logits, ctx = self.forward(X_train[idx])
                grads = self.backward(logits, y_train[idx], ctx)
                self._clip(grads)
                self._adam_step(grads, lr)
            dev_acc = self.accuracy(X_dev, y_dev)
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_params = {k: v.copy() for k, v in self.params.items()}
            print(f"  [manual] epoch {epoch + 1}/{epochs}  dev_acc={dev_acc:.3f}")
        if best_params is not None:
            self.params = best_params
        print(f"  [manual] best dev_acc={best_dev:.3f}")

    def save(self, path):
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path):
        model = cls()
        npz = np.load(path)
        model.params = {k: npz[k] for k in (*_GATES, "Wy", "by")}
        return model
