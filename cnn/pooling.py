"""#11 池化：朴素手写最大池化 / 平均池化，对照 nn.MaxPool2d。"""
import numpy as np


def maxpool2d_naive(x, k=2, stride=2):
    H, W = x.shape
    outH = (H - k) // stride + 1
    outW = (W - k) // stride + 1
    out = np.zeros((outH, outW))
    for i in range(outH):
        for j in range(outW):
            out[i, j] = np.max(x[i * stride:i * stride + k, j * stride:j * stride + k])
    return out


def avgpool2d_naive(x, k=2, stride=2):
    H, W = x.shape
    outH = (H - k) // stride + 1
    outW = (W - k) // stride + 1
    out = np.zeros((outH, outW))
    for i in range(outH):
        for j in range(outW):
            out[i, j] = np.mean(x[i * stride:i * stride + k, j * stride:j * stride + k])
    return out


def receptive_field(layers, kernel_size=3, stride=1, pool=2):
    """粗略估算叠加若干层后每个神经元的感受野大小（输入侧能看到多大区域）。"""
    rf = 1
    for _ in range(layers):
        rf = rf + (kernel_size - 1) * stride * (2 if False else 1)  # 简化示意
        rf = rf * pool  # 每层后接一次池化，感受野翻倍
    return rf
