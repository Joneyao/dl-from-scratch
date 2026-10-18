"""教学用数据集生成。"""
import numpy as np


def make_concentric_data(n=300, noise=0.12, seed=0):
    """同心圆二分类：内圆是类别 1，外环是类别 0，线性不可分。"""
    rng = np.random.default_rng(seed)
    n_inner = n // 2
    n_outer = n - n_inner
    theta_in = rng.uniform(0, 2 * np.pi, n_inner)
    r_in = 0.5 + rng.normal(0, noise, n_inner)
    inner = np.stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)], axis=1)
    theta_out = rng.uniform(0, 2 * np.pi, n_outer)
    r_out = 1.5 + rng.normal(0, noise, n_outer)
    outer = np.stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)], axis=1)
    X = np.vstack([inner, outer]).astype(np.float32)
    y = np.hstack([np.ones(n_inner), np.zeros(n_outer)]).astype(np.float32)
    return X, y


def make_spirals(n=400, turns=1.5, noise=0.12, seed=0):
    """两条螺旋二分类：教科书级"深度比宽度更有效"的数据集，浅层网络极难分开。"""
    rng = np.random.default_rng(seed)
    n_per = n // 2
    theta = np.linspace(0, turns * 2 * np.pi, n_per)
    r = np.linspace(0, 1, n_per)
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    x2 = -r * np.cos(theta)
    y2 = -r * np.sin(theta)
    X = np.vstack([np.stack([x1, y1], axis=1), np.stack([x2, y2], axis=1)]).astype(np.float32)
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    y = np.hstack([np.zeros(n_per), np.ones(n_per)]).astype(np.float32)
    return X, y


def make_linear_data(n=50, slope=2.0, intercept=1.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.linspace(-3, 3, n).astype(np.float32)
    y = (slope * X + intercept + rng.normal(0, noise, size=n)).astype(np.float32)
    return X, y
