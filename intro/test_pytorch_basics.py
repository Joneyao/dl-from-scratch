import numpy as np
from intro.pytorch_basics import manual_gradient_demo, linear_regression_torch
from common.datasets import make_linear_data


def test_manual_gradient_matches_chain_rule():
    # y = x^2 + 2x + 1, dy/dx = 2x + 2 = 8 at x=3
    assert abs(manual_gradient_demo() - 8.0) < 1e-5


def test_linear_regression_recovers_params():
    X, y = make_linear_data(n=80, slope=2.0, intercept=1.0, noise=0.0, seed=2)
    w, b, history = linear_regression_torch(X, y, lr=0.05, n_iters=500)
    assert abs(w - 2.0) < 0.05 and abs(b - 1.0) < 0.05
    assert history[-1] < history[0]
