# Activation Gradient Flow: XOR Activation Experiments

## Report

The full technical report is available here:

[Activation Gradient Flow: XOR Activation Experiments](paper/activation-gradient-flow-xor-activation-study.pdf)

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

Each activation/scale setting is repeated across 10 random seeds. The code logs train/test performance, derivative diagnostics, pre-activation diagnostics, activation-limit proxies, and gradient statistics.

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
```

Generated experiment CSVs are written to `results/`, which is intentionally ignored by Git.

Draft figures are written to `figures_drafts/`, which is also ignored by Git.

The final report figures are stored in `figures_final/`.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/MBELSABAH/activation-gradient-flow.git
cd activation-gradient-flow
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Smoke test

Run a small test to verify the training and plotting pipeline:

```bash
python3 src/train.py --run-tag smoke --seeds 0 --epochs 50 \
  --scales 1.0 --inits xavier --activations relu,tanh

python3 src/plot_results.py --run-tag smoke --init xavier \
  --scale 1.0 --metric test_acc \
  --save figures_drafts/smoke_test_acc.png
```

Expected outputs:

```text
results/final_smoke.csv
results/summary_smoke_relu_xavier_s1.csv
results/summary_smoke_tanh_xavier_s1.csv
figures_drafts/smoke_test_acc.png
```

---

## Final Xavier sweep used in the report

The main experiment in the report uses Xavier initialization with scales `0.5`, `1.0`, and `2.0`, repeated across 10 seeds.

```bash
python3 src/train.py --run-tag sweep_xavier --inits xavier \
  --scales 0.5,1.0,2.0 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --epochs 3000 \
  --activations relu,tanh,arctan,softsign
```

This produces:

```text
results/run_sweep_xavier_<activation>_xavier_<scale>_seed<seed>.csv
results/summary_sweep_xavier_<activation>_xavier_<scale>.csv
results/final_sweep_xavier.csv
```

Example files:

```text
results/final_sweep_xavier.csv
results/summary_sweep_xavier_relu_xavier_s0p5.csv
results/summary_sweep_xavier_tanh_xavier_s1.csv
results/summary_sweep_xavier_arctan_xavier_s2.csv
```

---

## Final figures used in the report

Generate the final report figures:

```bash
mkdir -p figures_final

python3 src/plot_results.py --run-tag sweep_xavier --init xavier \
  --scale 0.5 --metric test_acc \
  --save figures_final/xavier_s0p5_test_acc.png

python3 src/plot_results.py --run-tag sweep_xavier --init xavier \
  --scale 0.5 --metric z_near0_rate \
  --y-min 0.15 --y-max 0.85 \
  --save figures_final/xavier_s0p5_z_near0_rate.png

python3 src/plot_results.py --run-tag sweep_xavier --init xavier \
  --scale 0.5 --metric z_p90_abs \
  --save figures_final/xavier_s0p5_z_p90_abs.png

python3 src/plot_results.py --run-tag sweep_xavier --init xavier \
  --scale 2.0 --metric test_acc \
  --save figures_final/xavier_s2_test_acc.png

python3 src/plot_results.py --run-tag sweep_xavier --init xavier \
  --scale 2.0 --metric dphi_p01_abs \
  --activations tanh,arctan,softsign \
  --save figures_final/xavier_s2_dphi_p01_smooth.png
```

The expected final figures are:

```text
figures_final/xavier_s0p5_test_acc.png
figures_final/xavier_s0p5_z_near0_rate.png
figures_final/xavier_s0p5_z_p90_abs.png
figures_final/xavier_s2_test_acc.png
figures_final/xavier_s2_dphi_p01_smooth.png
```

---

## Key command-line options

`src/train.py` supports:

```text
--run-tag
--activations
--inits
--scales
--seeds
--n
--noise
--test-size
--lr
--epochs
--arch
--deriv-eps
--z-linear-eps
--relabel-after-noise
```

Example custom run:

```bash
python3 src/train.py --run-tag custom_run \
  --inits xavier \
  --scales 1.0 \
  --seeds 0,1,2 \
  --epochs 1000 \
  --activations relu,tanh,arctan,softsign
```

---

## Plotting arbitrary metrics

`src/plot_results.py` plots mean ± standard deviation curves from aggregated summary CSVs.

General format:

```bash
python3 src/plot_results.py --run-tag <run_tag> \
  --init <init_name> \
  --scale <scale_value> \
  --metric <metric_name> \
  --save <output_path>
```

Example:

```bash
python3 src/plot_results.py --run-tag sweep_xavier \
  --init xavier \
  --scale 1.0 \
  --metric test_acc \
  --save figures_drafts/xavier_s1_test_acc.png
```

Zooming the y-axis:

```bash
python3 src/plot_results.py --run-tag sweep_xavier \
  --init xavier \
  --scale 0.5 \
  --metric z_near0_rate \
  --y-min 0.15 --y-max 0.85 \
  --save figures_drafts/xavier_s0p5_z_near0_rate_zoomed.png
```

---

## Logged diagnostics

For every epoch, the code records the following metrics.

### Performance metrics

```text
train_loss
test_loss
train_acc
test_acc
```

### Derivative diagnostics

```text
dphi_small_rate
dphi_mean_abs
dphi_p10_abs
dphi_p01_abs
dphi_log_mean
```

These summarize the magnitude of hidden-layer activation derivatives.

### Pre-activation diagnostics

```text
z_mean_abs
z_p90_abs
z_near0_rate
```

These describe the scale of hidden pre-activations.

### Activation-limit proxy

```text
limit_rate
```

This is an activation-specific qualitative diagnostic:

- ReLU: fraction of hidden pre-activations with `z <= 0`
- tanh: fraction with `|tanh(z)| > 0.95`
- arctan: fraction with `|arctan(z)| > 0.95 * pi/2`
- softsign: fraction with `|softsign(z)| > 0.95`

### Gradient statistics

```text
grad_norm_L1
grad_norm_L2
grad_norm_L3
grad_rms_L1
grad_rms_L2
grad_rms_L3
```

These summarize layerwise weight-gradient norms and RMS values.

---

## Output files

Each run writes three categories of output.

### Per-seed logs

```text
results/run_<tag>_<activation>_<init>_<scale_tag>_seed<seed>.csv
```

Example:

```text
results/run_sweep_xavier_relu_xavier_s0p5_seed0.csv
```

### Aggregated summary logs

```text
results/summary_<tag>_<activation>_<init>_<scale_tag>.csv
```

Example:

```text
results/summary_sweep_xavier_relu_xavier_s0p5.csv
```

These contain mean and standard deviation per epoch across seeds.

### Final summary table

```text
results/final_<tag>.csv
```

Example:

```text
results/final_sweep_xavier.csv
```

This contains final-epoch mean and standard deviation values for each activation/init/scale setting.

---

## Notes on generated files

The following are ignored by Git:

```text
results/
figures_drafts/
__pycache__/
*.pyc
.DS_Store
archive_old_outputs/
```

The final figures in `figures_final/` are intended to be committed because they are used by the report.

---

## Main result table

Final test accuracy from the Xavier sweep:

| Activation | Scale 0.5 | Scale 1.0 | Scale 2.0 |
|---|---:|---:|---:|
| ReLU | 0.9067 ± 0.0260 | 0.9080 ± 0.0276 | 0.9120 ± 0.0247 |
| tanh | 0.7573 ± 0.1431 | 0.9093 ± 0.0278 | 0.9107 ± 0.0231 |
| arctan | 0.7000 ± 0.1503 | 0.9107 ± 0.0304 | 0.9107 ± 0.0239 |
| softsign | 0.8640 ± 0.0605 | 0.9053 ± 0.0227 | 0.9107 ± 0.0231 |

Interpretation:

- At scale `0.5`, ReLU is substantially more reliable.
- At scales `1.0` and `2.0`, all activations reach similar final test accuracy.
- Diagnostic metrics reveal internal differences even when endpoint accuracy is similar.

---

## References

The experiment is motivated by standard results on backpropagation, Xavier/Glorot initialization, and ReLU-style activation functions:

- Goodfellow, Bengio, and Courville, *Deep Learning*, 2016.
- Glorot and Bengio, “Understanding the difficulty of training deep feedforward neural networks,” AISTATS 2010.
- Nair and Hinton, “Rectified linear units improve restricted Boltzmann machines,” ICML 2010.
- Rumelhart, Hinton, and Williams, “Learning representations by back-propagating errors,” *Nature*, 1986.
