"""TensorFlow/Keras version of the stacked GRU.

`KerasGRU` mirrors the from-scratch `gru_scratch.GRU`: same constructor inputs and the
same `train()` / `predict()` interface, so `utils.py` and `compare.py` drive it unchanged.
It is standalone (a stack of `keras.layers.GRU` trained with Adam) and does not share
weights with the manual model.
"""

from tensorflow import keras


class KerasGRU:

    def __init__(self, X, Y, hidden_layers=(100,), learning_rate=0.01,
                 epochs=15, batch_size=32, task="classification"):
        # X/Y are batch-first: (m, T_x, n_x) and (m, T_x, n_y).
        self.X = X
        self.Y = Y
        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.n_x = X.shape[2]
        self.n_y = Y.shape[2]
        self.model = None

    def _build_model(self, T_x):
        """Build the stacked-GRU + dense model for sequences of length T_x."""
        model = keras.Sequential()
        model.add(keras.Input(shape=(T_x, self.n_x)))

        # One return_sequences=True GRU per entry in hidden_layers, so a prediction
        # is produced at every timestep (many-to-many), matching the manual model.
        for units in self.hidden_layers:
            model.add(keras.layers.GRU(units, return_sequences=True))

        # Softmax over the vocabulary for classification, linear for regression.
        out_act = "softmax" if self.task == "classification" else "linear"
        model.add(keras.layers.Dense(self.n_y, activation=out_act))

        loss = ("categorical_crossentropy" if self.task == "classification"
                else "mse")
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                      loss=loss)
        return model

    def train(self):
        self.model = self._build_model(self.X.shape[1])
        # Mini-batch training over epochs (Keras shuffles batches internally).
        self.model.fit(self.X, self.Y, epochs=self.epochs,
                       batch_size=self.batch_size, verbose=0)
        final_loss = self.model.evaluate(self.X, self.Y, verbose=0)
        print(f"[Keras] trained {self.epochs} epochs (batch_size={self.batch_size}), "
              f"final loss {final_loss:.4f}")
        return self

    def predict(self, X):
        """Run the trained Keras model forward. Returns (m, T_x, n_y)."""
        return self.model.predict(X, verbose=0)
