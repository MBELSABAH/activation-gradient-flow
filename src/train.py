import argparse
import csv
import os

import numpy as np

from data import make_xor
from losses import binary_cross_entropy
from metrics import activation_diagnostics, gradient_stats
from model import MLP


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

FINAL_METRICS = [
    "train_loss",
    "test_loss",
    "train_acc",
    "test_acc",
    "dphi_small_rate",
    "dphi_mean_abs",
    "dphi_p10_abs",
    "dphi_p01_abs",
    "dphi_log_mean",
    "z_mean_abs",
    "z_p90_abs",
    "z_near0_rate",
    "limit_rate",
]


def parse_csv_arg(value, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def scale_tag(scale):
    text = f"{scale:g}"
    return f"s{text.replace('.', 'p')}"


def train_test_split(X, y, test_size, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def run_seed(args, activation, init_mode, init_scale, seed):
    X, y = make_xor(n=args.n, noise=args.noise, seed=seed, relabel_after_noise=args.relabel_after_noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, args.test_size, seed)
    model = MLP(args.arch, hidden_activation=activation, init_mode=init_mode, init_scale=init_scale, seed=seed)

    rows = []
    for epoch in range(args.epochs):
        yhat_train = model.forward(X_train)
        train_loss = float(binary_cross_entropy(y_train, yhat_train))
        train_acc = float(np.mean((yhat_train >= 0.5) == y_train))

        dW, db = model.backward(y_train)
        gstats = gradient_stats(dW)
        ad = activation_diagnostics(activation, model.hidden_preactivations(), args.deriv_eps, args.z_linear_eps)

        yhat_test = model.forward(X_test)
        test_loss = float(binary_cross_entropy(y_test, yhat_test))
        test_acc = float(np.mean((yhat_test >= 0.5) == y_test))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            **ad,
            **gstats,
        }
        rows.append(row)
        model.step(dW, db, args.lr)
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(seed_runs):
    keys = list(seed_runs[0][0].keys())
    out = []
    for epoch_idx in range(len(seed_runs[0])):
        row = {"epoch": seed_runs[0][epoch_idx]["epoch"]}
        for k in keys:
            if k == "epoch":
                continue
            vals = np.array([run[epoch_idx][k] for run in seed_runs], dtype=float)
            row[f"{k}_mean"] = float(vals.mean())
            row[f"{k}_std"] = float(vals.std(ddof=0))
        out.append(row)
    return out


def final_summary_row(activation, init_mode, scale, args, last):
    row = {
        "activation": activation,
        "init": init_mode,
        "init_scale": scale,
        "n_seeds": len(args.seeds),
        "epochs": args.epochs,
        "lr": args.lr,
        "noise": args.noise,
        "test_size": args.test_size,
    }
    for metric in FINAL_METRICS:
        row[f"{metric}_mean"] = last[f"{metric}_mean"]
        row[f"{metric}_std"] = last[f"{metric}_std"]
    return row


def main():
    p = argparse.ArgumentParser(description="Run activation gradient-flow experiments on noisy XOR.")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--activations", default="relu,tanh,arctan,softsign")
    p.add_argument("--inits", default="xavier")
    p.add_argument("--scales", default="1.0")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--arch", default="2,16,16,1")
    p.add_argument("--deriv-eps", type=float, default=1e-3)
    p.add_argument("--z-linear-eps", type=float, default=0.1)
    p.add_argument("--relabel-after-noise", action="store_true")
    args = p.parse_args()

    args.activations = parse_csv_arg(args.activations, str)
    args.inits = parse_csv_arg(args.inits, str)
    args.scales = parse_csv_arg(args.scales, float)
    args.seeds = parse_csv_arg(args.seeds, int)
    args.arch = parse_csv_arg(args.arch, int)

    final_rows = []
    for activation in args.activations:
        for init_mode in args.inits:
            for scale in args.scales:
                tag = scale_tag(scale)
                seed_runs = []
                for seed in args.seeds:
                    rows = run_seed(args, activation, init_mode, scale, seed)
                    seed_runs.append(rows)
                    run_path = os.path.join(RESULTS_DIR, f"run_{args.run_tag}_{activation}_{init_mode}_{tag}_seed{seed}.csv")
                    write_csv(run_path, rows)

                summary = aggregate_rows(seed_runs)
                summary_path = os.path.join(RESULTS_DIR, f"summary_{args.run_tag}_{activation}_{init_mode}_{tag}.csv")
                write_csv(summary_path, summary)
                final_rows.append(final_summary_row(activation, init_mode, scale, args, summary[-1]))

    write_csv(os.path.join(RESULTS_DIR, f"final_{args.run_tag}.csv"), final_rows)


if __name__ == "__main__":
    main()
