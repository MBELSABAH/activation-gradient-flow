import numpy as np


def _stack(arrays):
    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate([a.reshape(-1) for a in arrays])


def limit_rate_from_z(hidden_activation, z_abs, limit_thresh=0.95):
    if z_abs.size == 0:
        return 0.0
    if hidden_activation == "relu":
        return float(np.mean(z_abs == 0.0))
    if hidden_activation in {"tanh", "softsign"}:
        bound = 1.0
    elif hidden_activation == "arctan":
        bound = np.pi / 2.0
    else:
        return 0.0
    return float(np.mean(np.abs(z_abs) >= (limit_thresh * bound)))


def activation_diagnostics(hidden_activation, hidden_z, deriv_eps=1e-3, z_linear_eps=0.1):
    z_flat = _stack(hidden_z)
    if z_flat.size == 0:
        return {
            "dphi_small_rate": 0.0,
            "dphi_mean_abs": 0.0,
            "dphi_p10_abs": 0.0,
            "dphi_p01_abs": 0.0,
            "dphi_log_mean": -27.631,
            "z_mean_abs": 0.0,
            "z_p90_abs": 0.0,
            "z_near0_rate": 0.0,
            "limit_rate": 0.0,
        }

    if hidden_activation == "relu":
        dphi = (z_flat > 0).astype(float)
    elif hidden_activation == "tanh":
        dphi = 1.0 - np.tanh(z_flat) ** 2
    elif hidden_activation == "arctan":
        dphi = 1.0 / (1.0 + z_flat ** 2)
    elif hidden_activation == "softsign":
        dphi = 1.0 / (1.0 + np.abs(z_flat)) ** 2
    else:
        raise ValueError(f"Unknown activation '{hidden_activation}'")

    dphi_abs = np.abs(dphi)
    z_abs = np.abs(z_flat)

    return {
        "dphi_small_rate": float(np.mean(dphi_abs < deriv_eps)),
        "dphi_mean_abs": float(np.mean(dphi_abs)),
        "dphi_p10_abs": float(np.percentile(dphi_abs, 10)),
        "dphi_p01_abs": float(np.percentile(dphi_abs, 1)),
        "dphi_log_mean": float(np.mean(np.log(dphi_abs + 1e-12))),
        "z_mean_abs": float(np.mean(z_abs)),
        "z_p90_abs": float(np.percentile(z_abs, 90)),
        "z_near0_rate": float(np.mean(z_abs <= z_linear_eps)),
        "limit_rate": float(np.mean(dphi_abs < 0.05)),
    }


def gradient_stats(dW):
    stats = {}
    for idx, dw in enumerate(dW, start=1):
        l2 = float(np.linalg.norm(dw))
        stats[f"grad_norm_L{idx}"] = l2
        stats[f"grad_rms_L{idx}"] = float(np.sqrt(np.mean(dw ** 2)))
    return stats
