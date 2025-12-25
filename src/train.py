import numpy as np
from data import make_xor
from model import MLP
from losses import binary_cross_entropy

X, y = make_xor(n=300, noise=0.1)

model = MLP([2, 16, 16, 1], hidden_activation="relu")
lr = 0.05
epochs = 2000

for epoch in range(epochs):
    y_hat = model.forward(X)
    loss = binary_cross_entropy(y, y_hat)

    dW, db = model.backward(y)
    model.step(dW, db, lr)

    if epoch % 200 == 0:
        acc = np.mean((y_hat > 0.5) == y)
        print(f"Epoch {epoch:4d} | Loss {loss:.4f} | Acc {acc:.3f}")
