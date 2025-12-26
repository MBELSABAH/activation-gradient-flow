import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)

RUN_TAG = "xavier"
activations = ["relu", "tanh", "arctan", "softsign"]

plt.figure(figsize=(10, 6))

for act in activations:
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"runs_{RUN_TAG}_{act}.csv"))
    plt.plot(df["epoch"], df["accuracy"], label=act)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch for Different Activations (Xavier init)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
