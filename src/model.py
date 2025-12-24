import numpy as np
from activations import sigmoid, ACTIVATIONS

class MLP:
    def __init__(self, layer_sizes, hidden_activation="relu", seed=42):
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []

        self.hidden_activation_name = hidden_activation
        self.hidden_activation, self.hidden_activation_derivative = \
            ACTIVATIONS[hidden_activation]

        for i in range(len(layer_sizes) - 1):
            self.W.append(
                rng.normal(0, 0.1, size=(layer_sizes[i], layer_sizes[i+1]))
            )
            self.b.append(np.zeros((1, layer_sizes[i+1])))

    def forward(self, X):
        self.Z = []
        self.A = [X]

        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.A[-1] @ W + b

            if i == len(self.W) - 1:
                a = sigmoid(z)                 # output layer fixed
            else:
                a = self.hidden_activation(z)  # controlled variable

            self.Z.append(z)
            self.A.append(a)

        return self.A[-1]

    def backward(self, y):
        m = y.shape[0]
        dW = []
        db = []

        dz = self.A[-1] - y   # sigmoid + BCE simplification

        for i in reversed(range(len(self.W))):
            dw = self.A[i].T @ dz / m
            db_ = np.mean(dz, axis=0, keepdims=True)

            dW.insert(0, dw)
            db.insert(0, db_)

            if i > 0:
                dz = (dz @ self.W[i].T) * \
                     self.hidden_activation_derivative(self.Z[i-1])

        return dW, db

    def step(self, dW, db, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]
