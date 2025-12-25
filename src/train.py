import os
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

# ---------- paths ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- experiment config ----------
activations = ["relu", "tanh", "arctan", "softsign"]
epochs = 2000
lr = 0.05

X, y = make_xor(n=300, noise=0.1)

# ---------- run experiments ----------
for activation in activations:
    print(f"\n=== Running activation: {activation} ===")

    model = MLP(
        [2, 16, 16, 1],
        hidden_activation=activation,
        seed=42
    )

    output_file = os.path.join(
        RESULTS_DIR, f"runs_{activation}.csv"
    )

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "loss",
            "accuracy",
            "grad_norms",
            "dead_or_saturation_rate"
        ])

        for epoch in range(epochs):
            y_hat = model.forward(X)
            loss = binary_cross_entropy(y, y_hat)

            dW, db = model.backward(y)
            model.step(dW, db, lr)

            acc = np.mean((y_hat > 0.5) == y)

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
                grad_norm,
                float(dead_rate)
            ])

            if epoch % 400 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Loss {loss:.4f} | "
                    f"Acc {acc:.3f} | "
                    f"Sat/Dead {dead_rate:.3f}"
                )

    print(f"Saved results to {output_file}")
