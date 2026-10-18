import torch
from training.init import build_net, layer_activation_std, he_init, xavier_init, compare_init


def test_layer_activation_std_length():
    net = build_net(5, he_init, torch.nn.ReLU())
    x = torch.randn(256, 256)
    stds = layer_activation_std(net, x)
    assert len(stds) == 6  # 输入 + 5 层输出


def test_he_init_keeps_signal_alive():
    # He 初始化 + ReLU：激活标准差应大致稳定在 1 附近，不衰减到 0
    torch.manual_seed(0)
    net = build_net(20, he_init, torch.nn.ReLU())
    x = torch.randn(256, 256)
    stds = layer_activation_std(net, x)
    assert stds[-1] > 0.1  # 20 层后信号仍存活
    assert stds[-1] < 10.0  # 也没有爆炸


def test_compare_init_returns_three():
    result = compare_init(depth=8, seed=0)
    assert set(result.keys()) == {"default", "xavier", "he"}
