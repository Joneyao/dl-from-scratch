import numpy as np
import torch
import torch.nn as nn
from cnn.conv import conv2d_naive, edge_kernel, detect_edges, blur


def test_conv_matches_torch():
    x = np.random.default_rng(0).normal(size=(8, 8))
    k = np.random.default_rng(1).normal(size=(3, 3))
    out_np = conv2d_naive(x, k, padding=1)
    xt = torch.tensor(x[None, None, :, :], dtype=torch.float32)
    conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor(k[None, None, :, :]))
        out_t = conv(xt).squeeze().numpy()
    assert np.allclose(out_np, out_t, atol=1e-5)


def test_edge_kernel_highlights_edges():
    # 一张左黑右白的图，中间有竖直边缘
    img = np.zeros((10, 10))
    img[:, 5:] = 1.0
    edges = detect_edges(img)
    # 边缘处的响应应该远大于平坦区域
    assert abs(edges[:, 5]).max() > abs(edges[:, 0]).max()


def test_blur_smooths():
    img = np.zeros((10, 10))
    img[4:6, 4:6] = 1.0  # 中间一个亮块
    blurred = blur(img)
    # 模糊后中心仍亮，但扩散到了周围
    assert blurred[4, 4] > 0 and blurred[3, 4] > 0
