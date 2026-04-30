import numpy as np

from activations import ACTIVATIONS, sigmoid


class MLP:
    def __init__(
        self,
        layer_sizes,
        hidden_activation="relu",
        init_mode="xavier",
        init_scale=1.0,
        seed=42,
    ):
        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{hidden_activation}'.")
        if init_mode not in {"xavier", "he", "gaussian"}:
            raise ValueError(f"Unknown init mode '{init_mode}'.")

        self.hidden_activation_name = hidden_activation
        self.hidden_activation, self.hidden_activation_derivative = ACTIVATIONS[hidden_activation]
        self.init_mode = init_mode
        self.init_scale = float(init_scale)

        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            if init_mode == "xavier":
                std = np.sqrt(2.0 / (fan_in + fan_out))
            elif init_mode == "he":
                std = np.sqrt(2.0 / fan_in)
            else:
                std = 0.1
            std *= self.init_scale

            self.W.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out)))

    def forward(self, X):
        self.Z = []
        self.A = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.A[-1] @ W + b
            if i == len(self.W) - 1:
                a = sigmoid(z)
            else:
                a = self.hidden_activation(z)
            self.Z.append(z)
            self.A.append(a)
        return self.A[-1]

    def backward(self, y):
        m = y.shape[0]
        dW = []
        db = []
        dz = self.A[-1] - y

        for i in reversed(range(len(self.W))):
            dw = self.A[i].T @ dz / m
            db_i = np.mean(dz, axis=0, keepdims=True)
            dW.insert(0, dw)
            db.insert(0, db_i)
            if i > 0:
                dz = (dz @ self.W[i].T) * self.hidden_activation_derivative(self.Z[i - 1])
        return dW, db

    def step(self, dW, db, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def hidden_preactivations(self):
        return self.Z[:-1]
