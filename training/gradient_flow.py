"""#04 梯度消失与梯度爆炸：深层网络里信号如何逐层衰减或放大。"""
import torch
import torch.nn as nn


def build_net(depth: int, activation: nn.Module) -> nn.Sequential:
    """搭一个 depth 个隐藏层的 8-宽全连接网络，逐层 Linear + 激活。"""
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(8, 8), activation]
    layers += [nn.Linear(8, 1)]
    return nn.Sequential(*layers)


def layer_grad_norms(net: nn.Sequential, x: torch.Tensor) -> list:
    """前向 + 反向，返回每个 Linear 层权重的梯度范数（衡量该层学到多少）。"""
    net.zero_grad()
    out = net(x)
    out.sum().backward()
    norms = []
    for layer in net:
        if isinstance(layer, nn.Linear):
            norms.append(float(layer.weight.grad.norm().item()))
    return norms


def compare_sigmoid_vs_relu(depth: int = 20, seed: int = 0) -> tuple:
    """返回 (sigmoid 各层梯度范数, relu 各层梯度范数)。"""
    torch.manual_seed(seed)
    x = torch.randn(8, 8) * 0.1
    sig_net = build_net(depth, nn.Sigmoid())
    relu_net = build_net(depth, nn.ReLU())
    return layer_grad_norms(sig_net, x), layer_grad_norms(relu_net, x)
