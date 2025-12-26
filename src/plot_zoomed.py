import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

RUN_TAG = "xavier"
activations = ["tanh", "arctan", "softsign"]

# ---- zoom window (late training) ----
X_MIN, X_MAX = 800, 2000  # focus on convergence region

plt.figure(figsize=(10, 6))

for act in activations:
    path = os.path.join(RESULTS_DIR, f"runs_{RUN_TAG}_{act}.csv")
    df = pd.read_csv(path)
    plt.plot(df["epoch"], df["accuracy"], label=act)

plt.xlim(X_MIN, X_MAX)
plt.ylim(0.84, 0.93)  # fixed y-range starting at 0.84

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Zoomed Accuracy Comparison (Smooth Activations, Xavier init)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
