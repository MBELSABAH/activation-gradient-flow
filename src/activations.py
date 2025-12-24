import numpy as np

# ---------- Sigmoid (output layer only) ----------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# ---------- Tanh ----------

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


# ---------- ReLU ----------

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)


# ---------- Leaky ReLU ----------

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


# ---------- Arctan ----------

def arctan(z):
    return np.arctan(z)

def arctan_derivative(z):
    return 1 / (1 + z ** 2)


# ---------- Softsign ----------

def softsign(z):
    return z / (1 + np.abs(z))

def softsign_derivative(z):
    return 1 / (1 + np.abs(z)) ** 2


# ---------- Activation registry ----------

ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
    "arctan": (arctan, arctan_derivative),
    "softsign": (softsign, softsign_derivative),
}
