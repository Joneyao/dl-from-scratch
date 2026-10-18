"""#05 激活函数：从 Sigmoid 到 ReLU 家族，前向与导数。"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    t = np.tanh(x)
    return 1.0 - t ** 2


def relu(x):
    return np.maximum(0.0, x)


def relu_deriv(x):
    return np.where(x > 0, 1.0, 0.0)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def gelu(x):
    """GELU 的 erf 近似：x·Φ(x) ≈ 0.5·x·(1+tanh(√(2/π)·(x+0.044715x³)))。"""
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def gelu_deriv(x):
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    sech = 1.0 / np.cosh(inner)
    return 0.5 * (1.0 + np.tanh(inner)) + 0.5 * x * sech ** 2 * c * (1.0 + 0.134145 * x ** 2)
