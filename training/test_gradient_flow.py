import numpy as np
import torch
from training.gradient_flow import build_net, layer_grad_norms, compare_sigmoid_vs_relu


def test_sigmoid_gradient_vanishes_deeper_than_relu():
    sig_norms, relu_norms = compare_sigmoid_vs_relu(depth=15, seed=0)
    # sigmoid 网络第一层梯度应远小于 relu 网络第一层（梯度消失）
    assert sig_norms[0] < relu_norms[0]
    # sigmoid 网络越靠前的层梯度越小（指数衰减），最后一层最大
    assert sig_norms[0] < sig_norms[-1]


def test_layer_grad_norms_count():
    net = build_net(5, torch.nn.ReLU())
    x = torch.randn(8, 8) * 0.1
    norms = layer_grad_norms(net, x)
    assert len(norms) == 6  # depth=5 个隐藏层 + 1 个输出层
