import numpy as np

def gradient_norms(dW):
    """
    Compute L2 norm of gradients for each layer.
    """
    return [np.linalg.norm(dw) for dw in dW]


def relu_dead_rate(activations):
    """
    Fraction of activations that are exactly zero.
    """
    a = np.concatenate(activations, axis=0)
    return np.mean(a == 0)


def sigmoid_saturation_rate(activations, eps=0.05):
    """
    Fraction of activations near 0 or 1.
    """
    a = np.concatenate(activations, axis=0)
    return np.mean((a < eps) | (a > 1 - eps))


def tanh_saturation_rate(activations, eps=0.95):
    """
    Fraction of activations near -1 or 1.
    """
    a = np.concatenate(activations, axis=0)
    return np.mean(np.abs(a) > eps)
