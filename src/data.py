import numpy as np

def make_xor(n=200, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)

    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] * X[:, 1] < 0).astype(int)

    if noise > 0:
        X += noise * rng.normal(size=X.shape)

    return X, y.reshape(-1, 1)
