"""#10 卷积：朴素手写二维卷积，对照 nn.Conv2d。"""
import numpy as np


def conv2d_naive(x, kernel, stride=1, padding=0):
    """朴素手写单通道二维卷积（滑动窗口，不追求性能，只为讲清原理）。

    x: (H, W) 单通道输入。kernel: (kH, kW) 卷积核。
    返回 (outH, outW) 特征图。
    """
    if padding:
        x = np.pad(x, padding, mode="constant")
    H, W = x.shape
    kH, kW = kernel.shape
    outH = (H - kH) // stride + 1
    outW = (W - kW) // stride + 1
    out = np.zeros((outH, outW))
    for i in range(outH):
        for j in range(outW):
            region = x[i * stride:i * stride + kH, j * stride:j * stride + kW]
            out[i, j] = float(np.sum(region * kernel))
    return out


def edge_kernel():
    """一个 3x3 边缘检测卷积核（拉普拉斯算子），能突出图像里的边缘。"""
    return np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)


def blur_kernel():
    """一个 3x3 均值模糊卷积核（权重共享的直观例子）。"""
    return np.ones((3, 3), dtype=float) / 9.0


def detect_edges(image):
    """对灰度图做边缘检测，返回边缘强度图。"""
    return conv2d_naive(image, edge_kernel(), padding=1)


def blur(image):
    return conv2d_naive(image, blur_kernel(), padding=1)
