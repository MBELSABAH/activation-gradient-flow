# Activation Gradient Flow: XOR Activation Experiments

This repository contains a from-scratch NumPy implementation of a shallow multilayer perceptron (MLP) used to study how activation functions interact with initialization scale, gradient flow, and training dynamics on a noisy XOR classification task.

The project compares:

- ReLU
- tanh
- arctan
- softsign

across Xavier/Glorot initialization scales:

- 0.5
- 1.0
- 2.0

Each activation/scale setting is repeated across 10 random seeds, and the code logs train/test performance, derivative diagnostics, pre-activation diagnostics, activation-limit proxies, and gradient statistics.

---

## Project summary

The main result is that activation comparisons are not stable under a single initialization scale.

At small Xavier scale (`0.5`), ReLU learns reliably while bounded smooth activations, especially tanh and arctan, learn more slowly and show higher seed-to-seed variance.

At moderate and larger Xavier scales (`1.0` and `2.0`), all four activations reach similar final test accuracy, but internal diagnostics still reveal different optimization behavior. In particular, derivative-tail metrics show that tanh develops a much smaller low-derivative tail than arctan and softsign at scale `2.0`.

The main conclusion is:

> Activation choice should be evaluated jointly with initialization scale, convergence dynamics, and gradient-flow diagnostics rather than endpoint accuracy alone.

---

## Repository structure

```text
activation-gradient-flow/
├── src/
│   ├── activations.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── plot_results.py
│   ├── plot_zoomed.py
│   └── train.py
├── figures_final/
│   ├── xavier_s0p5_test_acc.png
│   ├── xavier_s0p5_z_near0_rate.png
│   ├── xavier_s0p5_z_p90_abs.png
│   ├── xavier_s2_test_acc.png
│   └── xavier_s2_dphi_p01_smooth.png
├── requirements.txt
└── README.md
