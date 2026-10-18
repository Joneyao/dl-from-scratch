"""#07 归一化：手写 BatchNorm，训练/推理两种模式。"""
import numpy as np


def batchnorm_forward(x, gamma, beta, eps=1e-5, mode="train", running_mean=None, running_var=None, momentum=0.9):
    """手写 BatchNorm 前向。

    x: (N, D) 一个 batch。
    train 模式：用当前 batch 的均值方差归一化，并更新 running 统计。
    eval 模式：用 running 统计归一化（推理时没有 batch）。
    返回 (归一化输出, 更新后的 running_mean, running_var)。
    """
    if mode == "train":
        mean = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        if running_mean is None:
            running_mean = mean.squeeze().copy()
            running_var = var.squeeze().copy()
        else:
            running_mean = momentum * running_mean + (1 - momentum) * mean.squeeze()
            running_var = momentum * running_var + (1 - momentum) * var.squeeze()
    else:
        assert running_mean is not None, "eval 模式需要 running 统计"
        mean = running_mean.reshape(1, -1)
        var = running_var.reshape(1, -1)

    x_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    return out, running_mean, running_var


def train_torch_with_without_bn(X, y, use_bn, lr=0.1, n_iters=1000, seed=0):
    """用 PyTorch 对比加不加 BatchNorm1d 的训练收敛（返回损失历史）。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    D = X.shape[1]
    if use_bn:
        model = nn.Sequential(nn.Linear(D, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 1))
    else:
        model = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for _ in range(n_iters):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return history
