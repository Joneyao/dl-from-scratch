"""#03 PyTorch 三板斧：张量、autograd、nn.Module + 最小训练循环。"""
import torch
import torch.nn as nn


def manual_gradient_demo() -> float:
    """手推链式法则，对照 autograd：y = x^2 + 2x + 1, dy/dx = 2x + 2。"""
    x = torch.tensor(3.0, requires_grad=True)
    y = x * x + 2 * x + 1
    y.backward()
    return float(x.grad)


def linear_regression_torch(X, y, lr=0.05, n_iters=500):
    """用 autograd 手动实现 w、b 的梯度下降（对照 nn.Module 自动版）。"""
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    history = []
    for _ in range(n_iters):
        loss = torch.mean((w * Xt + b - yt) ** 2)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        history.append(float(loss.item()))
    return float(w.item()), float(b.item()), history


def nn_module_demo():
    """nn.Linear 的 forward + 参数。"""
    lin = nn.Linear(2, 1)
    x = torch.tensor([[1.0, 2.0]])
    return float(lin(x).item()), list(lin.parameters())
