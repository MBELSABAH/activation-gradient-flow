import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def scale_tag(scale):
    return f"s{f'{scale:g}'.replace('.', 'p')}"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float_array(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def main():
    p = argparse.ArgumentParser(description="Plot mean±std curves from summary CSVs.")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--activations", default="relu,tanh,arctan,softsign")
    p.add_argument("--x-min", type=float)
    p.add_argument("--x-max", type=float)
    p.add_argument("--y-min", type=float)
    p.add_argument("--y-max", type=float)
    p.add_argument("--save")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    acts = [a.strip() for a in args.activations.split(",") if a.strip()]
    tag = scale_tag(args.scale)

    plt.figure(figsize=(10, 6))
    plotted = 0
    for act in acts:
        path = os.path.join(RESULTS_DIR, f"summary_{args.run_tag}_{act}_{args.init}_{tag}.csv")
        if not os.path.exists(path):
            continue
        rows = read_csv(path)
        mean_key = f"{args.metric}_mean"
        std_key = f"{args.metric}_std"
        if not rows or mean_key not in rows[0] or std_key not in rows[0]:
            raise KeyError(f"Missing columns {mean_key}/{std_key} in {path}")
        x = to_float_array(rows, "epoch")
        y = to_float_array(rows, mean_key)
        s = to_float_array(rows, std_key)
        lower, upper = y - s, y + s
        if "acc" in args.metric:
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)
        elif np.min(y) >= 0.0:
            lower = np.clip(lower, 0.0, None)

        plt.plot(x, y, label=act)
        plt.fill_between(x, lower, upper, alpha=0.2)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No curves were plotted.")

    plt.xlabel("Epoch")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric}: {args.init}, scale={args.scale:g}, run={args.run_tag}")
    if args.x_min is not None or args.x_max is not None:
        plt.xlim(left=args.x_min, right=args.x_max)
    if args.y_min is not None or args.y_max is not None:
        plt.ylim(bottom=args.y_min, top=args.y_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.save:
        save_dir = os.path.dirname(args.save)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(args.save, dpi=args.dpi)
    else:
        plt.show()


if __name__ == "__main__":
    main()
