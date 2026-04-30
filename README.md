# Activation Gradient Flow (Noisy XOR)

From-scratch NumPy MLP experiments for ReLU, tanh, arctan, and softsign with initialization/scale sweeps and gradient-flow diagnostics.

## Quick smoke test

```bash
python src/train.py --run-tag smoke --seeds 0 --epochs 50 --scales 1.0 --inits xavier --activations relu,tanh
python src/plot_results.py --run-tag smoke --init xavier --scale 1.0 --metric test_acc --save figures_drafts/smoke_test_acc.png
```

## Recommended full sweeps

```bash
python src/train.py --run-tag sweep_xavier --inits xavier --scales 0.5,1.0,2.0 --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 3000 --activations relu,tanh,arctan,softsign
python src/train.py --run-tag sweep_he --inits he --scales 0.5,1.0,2.0 --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 3000 --activations relu,tanh,arctan,softsign
```

## Recommended figures

```bash
python src/plot_results.py --run-tag sweep_xavier --init xavier --scale 0.5 --metric test_acc --save figures_final/xavier_s0p5_test_acc.png
python src/plot_results.py --run-tag sweep_xavier --init xavier --scale 0.5 --metric z_near0_rate --y-min 0.85 --y-max 1.01 --save figures_final/xavier_s0p5_z_near0_rate.png
python src/plot_results.py --run-tag sweep_xavier --init xavier --scale 0.5 --metric z_p90_abs --save figures_final/xavier_s0p5_z_p90_abs.png
python src/plot_results.py --run-tag sweep_xavier --init xavier --scale 2.0 --metric test_acc --save figures_final/xavier_s2_test_acc.png
python src/plot_results.py --run-tag sweep_xavier --init xavier --scale 2.0 --metric dphi_p01_abs --activations tanh,arctan,softsign --save figures_final/xavier_s2_dphi_p01_smooth.png
```

## Outputs

- `results/run_<tag>_<activation>_<init>_<scale_tag>_seed<seed>.csv`
- `results/summary_<tag>_<activation>_<init>_<scale_tag>.csv`
- `results/final_<tag>.csv`
