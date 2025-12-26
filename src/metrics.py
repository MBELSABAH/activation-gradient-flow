import numpy as np

def gradient_norms(dW):
    """
    Compute L2 norm of gradients for each layer.
    """
    return [np.linalg.norm(dw) for dw in dW]


def _flatten_hidden_activations(activations):
    """
    Stack hidden-layer activations into one array for rate computations.
    activations is a list of arrays, one per hidden layer, each shape (n, units).
    """
    return np.concatenate(activations, axis=0)


def relu_dead_rate(activations):
    """
    Fraction of activations that are exactly zero.
    (For ReLU, exactly-zero activations are a reasonable "dead" proxy.)
    """
    a = _flatten_hidden_activations(activations)
    return np.mean(a == 0)


def sigmoid_saturation_rate(activations, eps=0.05):
    """
    Fraction of activations near 0 or 1.
    (Meaningful only for sigmoid-range activations.)
    """
    a = _flatten_hidden_activations(activations)
    return np.mean((a < eps) | (a > 1 - eps))


def tanh_saturation_rate(activations, eps=0.95):
    """
    Fraction of activations near -1 or 1.
    """
    a = _flatten_hidden_activations(activations)
    return np.mean(np.abs(a) > eps)


def arctan_saturation_rate(activations, eps=0.95):
    """
    Fraction of activations near the arctan bounds.
    arctan(z) ranges in (-pi/2, pi/2).
    We define "saturated" as close to either bound.
    """
    a = _flatten_hidden_activations(activations)
    bound = np.pi / 2
    thresh = eps * bound
    return np.mean((a < -thresh) | (a > thresh))


def softsign_saturation_rate(activations, eps=0.95):
    """
    Fraction of activations near the softsign bounds.
    softsign(z) = z / (1 + |z|) ranges in (-1, 1).
    """
    a = _flatten_hidden_activations(activations)
    return np.mean(np.abs(a) > eps)
