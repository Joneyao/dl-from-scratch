"""#09 正则化：手写 Dropout、weight decay，对比过拟合。"""
import numpy as np


def dropout_forward(x, p=0.5, training=True, seed=None):
    """手写 Dropout：训练时以概率 p 随机置零，并除以 1-p 保持期望不变。"""
    if not training or p <= 0.0:
        return x
    rng = np.random.default_rng(seed)
    mask = (rng.random(x.shape) > p).astype(float) / (1.0 - p)
    return x * mask


def make_poly_data(n=30, noise=0.3, seed=0):
    """真实关系是三次多项式，用来演示过拟合。"""
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-1, 1, size=n))
    y_true = 1.0 - 2.0 * X + 0.5 * X ** 2 + 1.5 * X ** 3
    y = y_true + rng.normal(0, noise, size=n)
    return X, y


def poly_features(X, degree):
    return np.vstack([X ** d for d in range(1, degree + 1)]).T


def fit_ridge(Phi, y, alpha):
    """岭回归（L2 weight decay）闭式解，返回 (coef, bias)。"""
    n, d = Phi.shape
    Xb = np.hstack([np.ones((n, 1)), Phi])
    reg = alpha * np.eye(d + 1)
    reg[0, 0] = 0.0
    theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return theta[1:], theta[0]


def predict_poly(coef, bias, X, degree):
    return poly_features(X, degree) @ coef + bias


def train_test_errors(degree, alpha, seed=0):
    Xtr, ytr = make_poly_data(n=30, seed=seed)
    Xte, yte = make_poly_data(n=30, seed=seed + 100)
    coef, bias = fit_ridge(poly_features(Xtr, degree), ytr, alpha)
    tr = float(np.mean((predict_poly(coef, bias, Xtr, degree) - ytr) ** 2))
    te = float(np.mean((predict_poly(coef, bias, Xte, degree) - yte) ** 2))
    return tr, te
