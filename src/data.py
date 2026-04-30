import numpy as np


def make_xor(n=300, noise=0.1, seed=42, relabel_after_noise=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] * X[:, 1] < 0).astype(int)
    if noise > 0:
        X = X + noise * rng.normal(size=X.shape)
        if relabel_after_noise:
            y = (X[:, 0] * X[:, 1] < 0).astype(int)
    return X, y.reshape(-1, 1)
