"""#01 浅层 vs 深层 MLP：在两条螺旋数据上对比，展示"深度"的力量。"""
import numpy as np
import torch
import torch.nn as nn


def shallow_net() -> nn.Module:
    """只有一个隐藏层的浅层网络（8 个神经元）。"""
    return nn.Sequential(
        nn.Linear(2, 8), nn.Tanh(),
        nn.Linear(8, 1),
    )


def deep_net() -> nn.Module:
    """三个隐藏层的深层网络（每层 8 个神经元，和浅层同样宽度）。"""
    return nn.Sequential(
        nn.Linear(2, 8), nn.Tanh(),
        nn.Linear(8, 8), nn.Tanh(),
        nn.Linear(8, 8), nn.Tanh(),
        nn.Linear(8, 1),
    )


def train(model: nn.Module, X, y, lr=0.01, n_iters=1200, seed=0) -> list:
    """Adam 训练，返回每轮损失历史。"""
    torch.manual_seed(seed)
    Xt = torch.tensor(X)
    yt = torch.tensor(y).view(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for _ in range(n_iters):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return history


def accuracy(model: nn.Module, X, y) -> float:
    Xt = torch.tensor(X)
    with torch.no_grad():
        pred = (model(Xt).squeeze() > 0).float().numpy()
    return float((pred == y).mean())


def decision_grid(model: nn.Module, xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), res=120):
    xs = np.linspace(xlim[0], xlim[1], res)
    ys = np.linspace(ylim[0], ylim[1], res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        z = torch.sigmoid(model(torch.tensor(pts))).squeeze().numpy().reshape(res, res)
    return gx, gy, z
