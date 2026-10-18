import numpy as np
from training.regularization import dropout_forward, train_test_errors, poly_features


def test_dropout_zeros_expected_fraction():
    x = np.ones((1000, 100))
    out = dropout_forward(x, p=0.5, training=True, seed=0)
    zero_frac = float((out == 0).mean())
    assert abs(zero_frac - 0.5) < 0.05


def test_dropout_preserves_expectation():
    x = np.ones((10000, 1))
    out = dropout_forward(x, p=0.5, training=True, seed=0)
    assert abs(float(out.mean()) - 1.0) < 0.05


def test_dropout_eval_is_identity():
    x = np.random.default_rng(0).normal(size=(10, 10))
    assert np.allclose(dropout_forward(x, p=0.5, training=False), x)


def test_poly_features_shape():
    X = np.array([1.0, 2.0, 3.0])
    P = poly_features(X, degree=3)
    assert P.shape == (3, 3)
    assert np.allclose(P[:, 2], X ** 3)


def test_high_degree_overfits():
    tr, te = train_test_errors(degree=12, alpha=0.0, seed=0)
    assert tr < te  # 训练误差明显低于测试误差 = 过拟合


def test_some_regularization_helps():
    _, te_no = train_test_errors(degree=12, alpha=0.0, seed=0)
    best = min(train_test_errors(degree=12, alpha=a, seed=0)[1] for a in [0.001, 0.01, 0.1, 1.0])
    assert best < te_no
