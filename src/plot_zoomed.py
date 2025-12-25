import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

activations = ["tanh", "arctan", "softsign"]

plt.figure(figsize=(10, 6))

for act in activations:
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"runs_{act}.csv"))
    plt.plot(df["epoch"], df["accuracy"], label=act)

# ---- zoom window ----
plt.ylim(0.5, 0.7)
plt.xlim(1200, 2100)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Zoomed Accuracy Comparison (Smooth Activations)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
