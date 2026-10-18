import numpy as np
from training.normalization import batchnorm_forward, train_torch_with_without_bn


def test_batchnorm_train_normalizes():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, (100, 8))
    gamma = np.ones(8)
    beta = np.zeros(8)
    out, rm, rv = batchnorm_forward(x, gamma, beta, mode="train")
    assert abs(float(out.mean())) < 1e-6
    assert abs(float(out.var()) - 1.0) < 1e-2


def test_batchnorm_eval_uses_running():
    x = np.random.default_rng(0).normal(0, 1, (1000, 4))
    gamma = np.ones(4)
    beta = np.zeros(4)
    # eval 模式应该用 running_mean=5 而不是当前 batch 的均值 0
    out, _, _ = batchnorm_forward(x, gamma, beta, mode="eval",
                                  running_mean=np.full(4, 5.0), running_var=np.ones(4))
    assert abs(float(out.mean()) - (-5.0)) < 0.2


def test_train_with_bn_converges():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 4))
    y = (X[:, 0] * 2 + X[:, 1]).reshape(-1, 1).astype(np.float32)
    hist = train_torch_with_without_bn(X, y, use_bn=True, n_iters=500, seed=0)
    assert not np.isnan(hist[-1])
    assert hist[-1] < hist[0]
