import csv
import numpy as np
from data import make_xor
from model import MLP
from losses import binary_cross_entropy
from metrics import (
    gradient_norms,
    relu_dead_rate,
    sigmoid_saturation_rate,
    tanh_saturation_rate
)
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X, y = make_xor(n=300, noise=0.1)

activation = "relu"
model = MLP([2, 16, 16, 1], hidden_activation=activation)

lr = 0.05
epochs = 2000

with open(os.path.join(RESULTS_DIR, "runs.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "loss",
        "accuracy",
        "activation",
        "grad_norms",
        "dead_or_saturation_rate"
    ])

    for epoch in range(epochs):
        y_hat = model.forward(X)
        loss = binary_cross_entropy(y, y_hat)

        dW, db = model.backward(y)
        model.step(dW, db, lr)

        acc = np.mean((y_hat > 0.5) == y)

        # ---- metrics ----
        grad_norm = gradient_norms(dW)

        hidden_A = model.hidden_activations()

        if activation == "relu":
            dead_rate = relu_dead_rate(hidden_A)
        elif activation == "tanh":
            dead_rate = tanh_saturation_rate(hidden_A)
        else:
            dead_rate = sigmoid_saturation_rate(hidden_A)

        writer.writerow([
            epoch,
            float(loss),
            float(acc),
            activation,
            grad_norm,
            float(dead_rate)
        ])

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss {loss:.4f} | "
                f"Acc {acc:.3f} | "
                f"GradNorm {grad_norm} | "
                f"Sat/Dead {dead_rate:.3f}"
            )
