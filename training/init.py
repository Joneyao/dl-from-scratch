"""#06 权重初始化：Xavier / He 如何让信号逐层不衰减不爆炸。"""
import torch
import torch.nn as nn


def xavier_init(layer: nn.Linear):
    """Xavier/Glorot：方差 = 2/(fan_in + fan_out)，配合 tanh/sigmoid。"""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def he_init(layer: nn.Linear):
    """He/Kaiming：方差 = 2/fan_in，配合 ReLU。"""
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def build_net(depth: int, init_fn, activation: nn.Module, width: int = 256) -> nn.Sequential:
    layers = [nn.Linear(width, width)]
    for _ in range(depth - 1):
        layers += [activation, nn.Linear(width, width)]
    for layer in layers:
        if isinstance(layer, nn.Linear):
            init_fn(layer)
    return nn.Sequential(*layers)


def layer_activation_std(net: nn.Sequential, x: torch.Tensor) -> list:
    """前向传播，记录每个 Linear 层输出的标准差（衡量信号是衰减还是爆炸）。"""
    stds = [float(x.std().item())]
    for layer in net:
        x = layer(x)
        if isinstance(layer, nn.Linear):
            stds.append(float(x.std().item()))
    return stds


def compare_init(depth: int = 20, seed: int = 0):
    """对比默认初始化 vs Xavier vs He 的逐层激活标准差。"""
    torch.manual_seed(seed)
    x = torch.randn(256, 256)

    def default_init(layer):
        pass  # PyTorch Linear 默认 kaiming_uniform

    nets = {
        "default": build_net(depth, default_init, nn.ReLU()),
        "xavier": build_net(depth, xavier_init, nn.ReLU()),
        "he": build_net(depth, he_init, nn.ReLU()),
    }
    return {name: layer_activation_std(net, x) for name, net in nets.items()}
