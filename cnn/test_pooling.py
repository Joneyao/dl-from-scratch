import numpy as np
import torch
import torch.nn as nn
from cnn.pooling import maxpool2d_naive, avgpool2d_naive


def test_maxpool_matches_torch():
    x = np.random.default_rng(0).normal(size=(6, 6))
    out_np = maxpool2d_naive(x, k=2, stride=2)
    xt = torch.tensor(x[None, None, :, :], dtype=torch.float32)
    out_t = nn.MaxPool2d(2)(xt).squeeze().numpy()
    assert np.allclose(out_np, out_t)


def test_avgpool_matches_torch():
    x = np.random.default_rng(0).normal(size=(6, 6))
    out_np = avgpool2d_naive(x, k=2, stride=2)
    xt = torch.tensor(x[None, None, :, :], dtype=torch.float32)
    out_t = nn.AvgPool2d(2)(xt).squeeze().numpy()
    assert np.allclose(out_np, out_t)


def test_maxpool_shrinks():
    x = np.random.default_rng(0).normal(size=(28, 28))
    out = maxpool2d_naive(x, k=2, stride=2)
    assert out.shape == (14, 14)
