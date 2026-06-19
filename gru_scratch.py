"""From-scratch, multi-layer GRU in pure NumPy.

This is the reference implementation the README derives step by step. It uses the
**batch-first** layout `(m, T_x, n_x)` (examples are rows), so every gate is computed
as `sigmoid(z @ W.T + b)` with `z = [a_prev, x_t]` — the transpose of the textbook
column-vector form. The forward pass (`layer_forward` / `gru_forward`) implements the
GRU equations (update gate, reset gate, candidate, hidden-state interpolation); the
backward pass (`layer_backward` / `gru_backward`) implements the batch-first BPTT
gradients from §7 of the README, line for line.

Tensor convention:
    m   : number of sequences in the batch
    T_x : time steps per sequence
    n_x : input feature size
    n_y : output feature size (vocab size)
    n_a : hidden state size of a layer
"""

import numpy as np


class GRU:

    def __init__(self, X, Y, hidden_layers=(100,),
                 learning_rate=0.01, epochs=15,
                 batch_size=32, task='classification'):

        self.X = X
        self.Y = Y

        self.hidden_layers = hidden_layers
        self.L = len(hidden_layers)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.task = task

        self.n_x = X.shape[-1]
        self.n_y = Y.shape[-1]

        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        # One set of gate weights per stacked layer. Convention (batch-first):
        #   a, x are (m, ...) row vectors; weights are (n_a, n_a + in_size);
        #   a gate is computed as  concat @ W.T + b  with b of shape (n_a,).
        # GRU has three weight matrices per layer: update gate (z), reset gate (r),
        # and candidate (h).
        Wz, bz, Wr, br, Wh, bh = [], [], [], [], [], []

        for l in range(self.L):
            n_a = self.hidden_layers[l]
            in_size = self.n_x if l == 0 else self.hidden_layers[l - 1]
            concat = n_a + in_size
            # Xavier/Glorot init: scaling by 1/sqrt(fan_in) keeps the gate
            # pre-activations in a sensible range so gradients neither vanish
            # nor explode -- a tiny fixed scale (e.g. 0.01) leaves the gates
            # almost frozen and the model barely learns under plain SGD.
            scale = 1.0 / np.sqrt(concat)
            Wz.append(np.random.randn(n_a, concat) * scale)
            bz.append(np.zeros(n_a))
            Wr.append(np.random.randn(n_a, concat) * scale)
            br.append(np.zeros(n_a))
            Wh.append(np.random.randn(n_a, concat) * scale)
            bh.append(np.zeros(n_a))

        # Output (dense) layer maps the top hidden state to the targets.
        Wy = np.random.randn(self.n_y, self.hidden_layers[-1]) / np.sqrt(self.hidden_layers[-1])
        by = np.zeros(self.n_y)

        return {
            'Wz': Wz, 'bz': bz,
            'Wr': Wr, 'br': br,
            'Wh': Wh, 'bh': bh,
            'Wy': Wy, 'by': by,
        }

    def layer_forward(self, X_seq, a0, l):

        Wz = self.parameters['Wz'][l]
        bz = self.parameters['bz'][l]
        Wr = self.parameters['Wr'][l]
        br = self.parameters['br'][l]
        Wh = self.parameters['Wh'][l]
        bh = self.parameters['bh'][l]

        m, T_x, _ = X_seq.shape
        n_a = self.hidden_layers[l]

        a_prev = a0

        a = np.zeros((m, T_x, n_a))

        layer_cache = []

        for t in range(T_x):
            xt = X_seq[:, t, :]                              # (m, in_size)
            concat = np.concatenate([a_prev, xt], axis=1)    # (m, n_a + in_size)

            zt = self.sigmoid(concat @ Wz.T + bz)            # update gate
            rt = self.sigmoid(concat @ Wr.T + br)            # reset gate

            # Candidate uses the *reset-gated* previous hidden state.
            concat_r = np.concatenate([rt * a_prev, xt], axis=1)  # (m, n_a + in_size)
            hh = np.tanh(concat_r @ Wh.T + bh)               # candidate hidden state

            # Interpolate between the old hidden state and the candidate.
            a_next = (1 - zt) * a_prev + zt * hh

            a[:, t, :] = a_next
            layer_cache.append((a_next, a_prev, zt, rt, hh, xt))

            a_prev = a_next

        return a, layer_cache

    def gru_forward(self, X, a0):

        caches_per_layer = []

        inp = X
        for l in range(self.L):
            a, layer_cache = self.layer_forward(inp, a0[l], l)
            caches_per_layer.append(layer_cache)
            inp = a                                          # feed states up the stack

        a_top = inp                                          # (m, T_x, n_a_top)

        # Many-to-many: a prediction at every timestep (next-token modelling).
        Wy = self.parameters['Wy']
        by = self.parameters['by']
        scores = a_top @ Wy.T + by                           # (m, T_x, n_y)

        if self.task == 'classification':
            y_pred = self.softmax(scores)
        else:
            y_pred = scores

        caches = (caches_per_layer, X, a_top)
        return caches, a_top, y_pred

    def layer_backward(self, da_above, layer_cache, l):

        Wz = self.parameters['Wz'][l]
        Wr = self.parameters['Wr'][l]
        Wh = self.parameters['Wh'][l]

        m, T_x, n_a = da_above.shape
        in_size = Wz.shape[1] - n_a

        dWz_l = np.zeros_like(Wz)
        dbz_l = np.zeros_like(self.parameters['bz'][l])
        dWr_l = np.zeros_like(Wr)
        dbr_l = np.zeros_like(self.parameters['br'][l])
        dWh_l = np.zeros_like(Wh)
        dbh_l = np.zeros_like(self.parameters['bh'][l])

        dx = np.zeros((m, T_x, in_size))
        da_next = np.zeros((m, n_a))

        for t in reversed(range(T_x)):

            (a_next, a_prev, zt, rt, hh, xt) = layer_cache[t]

            da = da_above[:, t, :] + da_next                  # total grad into a_next

            # Backprop through a_next = (1 - zt) * a_prev + zt * hh.
            dzt = da * (hh - a_prev)                          # into the update gate
            dhh = da * zt                                     # into the candidate
            da_prev = da * (1 - zt)                           # direct skip-connection path

            # Candidate hh = tanh(concat_r @ Wh.T + bh), concat_r = [rt * a_prev, xt].
            dhraw = dhh * (1 - hh ** 2)                        # pre-activation grad
            concat_r = np.concatenate([rt * a_prev, xt], axis=1)
            dWh_l += dhraw.T @ concat_r
            dbh_l += dhraw.sum(axis=0)

            dconcat_r = dhraw @ Wh                            # (m, n_a + in_size)
            d_rgated = dconcat_r[:, :n_a]                     # grad into (rt * a_prev)
            dx[:, t, :] += dconcat_r[:, n_a:]

            # Split the reset-gated term between the reset gate and a_prev.
            drt = d_rgated * a_prev
            da_prev += d_rgated * rt

            # Update gate zt = sigmoid(concat @ Wz.T + bz).
            dzraw = dzt * zt * (1 - zt)
            concat = np.concatenate([a_prev, xt], axis=1)
            dWz_l += dzraw.T @ concat
            dbz_l += dzraw.sum(axis=0)
            dconcat_z = dzraw @ Wz
            da_prev += dconcat_z[:, :n_a]
            dx[:, t, :] += dconcat_z[:, n_a:]

            # Reset gate rt = sigmoid(concat @ Wr.T + br).
            drraw = drt * rt * (1 - rt)
            dWr_l += drraw.T @ concat
            dbr_l += drraw.sum(axis=0)
            dconcat_r_gate = drraw @ Wr
            da_prev += dconcat_r_gate[:, :n_a]
            dx[:, t, :] += dconcat_r_gate[:, n_a:]

            da_next = da_prev

        grads = {
            'dWz_l': dWz_l, 'dbz_l': dbz_l,
            'dWr_l': dWr_l, 'dbr_l': dbr_l,
            'dWh_l': dWh_l, 'dbh_l': dbh_l,
        }
        return grads, dx

    def gru_backward(self, dscores, caches):

        (caches_per_layer, X, a_top) = caches

        # Output-layer gradients. dscores is d(loss)/d(scores) at every timestep,
        # e.g. (y_pred - Y) / (m * T_x) for softmax+cross-entropy or linear+MSE.
        Wy = self.parameters['Wy']
        dWy = np.einsum('mty,mta->ya', dscores, a_top)        # (n_y, n_a_top)
        dby = dscores.sum(axis=(0, 1))                        # (n_y,)
        da_top = dscores @ Wy                                 # (m, T_x, n_a_top)

        dWz = [None] * self.L
        dbz = [None] * self.L
        dWr = [None] * self.L
        dbr = [None] * self.L
        dWh = [None] * self.L
        dbh = [None] * self.L

        # The top layer receives the output-layer gradient at every timestep.
        da_above = da_top

        for l in reversed(range(self.L)):
            grads, dx = self.layer_backward(da_above, caches_per_layer[l], l)
            dWz[l] = grads['dWz_l']
            dbz[l] = grads['dbz_l']
            dWr[l] = grads['dWr_l']
            dbr[l] = grads['dbr_l']
            dWh[l] = grads['dWh_l']
            dbh[l] = grads['dbh_l']
            da_above = dx                                     # pass down the stack

        return {
            'dWz': dWz, 'dbz': dbz,
            'dWr': dWr, 'dbr': dbr,
            'dWh': dWh, 'dbh': dbh,
            'dWy': dWy, 'dby': dby,
        }

    def update_parameters(self, gradients):
        lr = self.learning_rate
        for l in range(self.L):
            self.parameters['Wz'][l] -= lr * gradients['dWz'][l]
            self.parameters['bz'][l] -= lr * gradients['dbz'][l]
            self.parameters['Wr'][l] -= lr * gradients['dWr'][l]
            self.parameters['br'][l] -= lr * gradients['dbr'][l]
            self.parameters['Wh'][l] -= lr * gradients['dWh'][l]
            self.parameters['bh'][l] -= lr * gradients['dbh'][l]
        self.parameters['Wy'] -= lr * gradients['dWy']
        self.parameters['by'] -= lr * gradients['dby']

    def compute_loss(self, y_pred, Y):
        # Averaged over both the batch (m) and the time axis (T_x), matching the
        # per-timestep next-token objective the framework wrappers also use.
        m, T_x = Y.shape[0], Y.shape[1]
        if self.task == 'classification':
            eps = 1e-12
            return -np.sum(Y * np.log(y_pred + eps)) / (m * T_x)
        return np.sum((y_pred - Y) ** 2) / (2 * m * T_x)

    def _clip_gradients(self, gradients, max_norm=5.0):
        # Global-norm gradient clipping keeps BPTT through long sequences stable.
        sq = 0.0
        for k, v in gradients.items():
            if isinstance(v, list):
                sq += sum(float(np.sum(g ** 2)) for g in v)
            else:
                sq += float(np.sum(v ** 2))
        norm = np.sqrt(sq)
        if norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            for k, v in gradients.items():
                if isinstance(v, list):
                    gradients[k] = [g * scale for g in v]
                else:
                    gradients[k] = v * scale
        return gradients

    def train(self):
        m = self.X.shape[0]
        batch_size = self.batch_size or m

        for epoch in range(self.epochs):
            perm = np.random.permutation(m)
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, m, batch_size):
                idx = perm[start:start + batch_size]
                Xb, Yb = self.X[idx], self.Y[idx]
                mb, T_x = Xb.shape[0], Xb.shape[1]

                a0 = [np.zeros((mb, n_a)) for n_a in self.hidden_layers]

                caches, a_top, y_pred = self.gru_forward(Xb, a0)

                epoch_loss += self.compute_loss(y_pred, Yb)
                n_batches += 1

                # Gradient of the (mean) loss w.r.t. the scores at every timestep:
                # (y_pred - Y) / (m * T_x) for softmax+CE and for linear+MSE.
                dscores = (y_pred - Yb) / (mb * T_x)

                gradients = self.gru_backward(dscores, caches)
                gradients = self._clip_gradients(gradients)
                self.update_parameters(gradients)

            print(f"epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss / max(n_batches, 1):.4f}")

        return self

    def predict(self, X):
        # Returns per-timestep probabilities, shape (m, T_x, n_y), as the shared
        # utils.evaluate / utils.generate helpers expect.
        m = X.shape[0]
        a0 = [np.zeros((m, n_a)) for n_a in self.hidden_layers]
        _, _, y_pred = self.gru_forward(X, a0)
        return y_pred

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        # Stable softmax over the last axis (works for 2-D and 3-D inputs).
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)
