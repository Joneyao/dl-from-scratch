import numpy as np
import torch
from intro.deep_vs_shallow import shallow_net, deep_net, train, accuracy, decision_grid
from common.datasets import make_spirals


def _count_linear(m):
    return sum(1 for layer in m if isinstance(layer, torch.nn.Linear))


def test_deep_net_has_more_layers():
    assert _count_linear(shallow_net()) == 2
    assert _count_linear(deep_net()) == 4


def test_train_reduces_loss():
    X, y = make_spirals(n=300, noise=0.1, seed=0)
    history = train(deep_net(), X, y, n_iters=300, seed=0)
    assert history[-1] < history[0]


def test_deep_beats_shallow_on_spirals():
    X, y = make_spirals(n=400, noise=0.1, seed=1)
    # 固定随机种子，保证模型初始化和训练都确定性可复现
    torch.manual_seed(0)
    shallow = shallow_net()
    torch.manual_seed(0)
    deep = deep_net()
    train(shallow, X, y, n_iters=1200, seed=0)
    train(deep, X, y, n_iters=1200, seed=0)
    acc_shallow = accuracy(shallow, X, y)
    acc_deep = accuracy(deep, X, y)
    assert acc_deep > acc_shallow + 0.03


def test_decision_grid_shape():
    X, _ = make_spirals(n=50, seed=0)
    model = deep_net()
    gx, gy, z = decision_grid(model, res=60)
    assert z.shape == (60, 60)
