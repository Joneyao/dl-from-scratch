"""#12 经典 CNN 架构：LeNet 与 VGG 风格网络。"""
import torch.nn as nn


def lenet(num_classes=10):
    """LeNet：卷积网络鼻祖，1998 年用于手写数字识别。"""
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, num_classes),
    )


def vgg_block(in_c, out_c, num_convs=2):
    """VGG 风格的块：连续几个 3x3 卷积 + ReLU，最后接一个池化。"""
    layers = []
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.ReLU()]
        in_c = out_c
    layers += [nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


def vgg_style(num_classes=10):
    """一个 VGG 风格的网络：只用 3x3 小卷积核，靠堆深。"""
    return nn.Sequential(
        vgg_block(1, 32, num_convs=2),
        vgg_block(32, 64, num_convs=2),
        vgg_block(64, 128, num_convs=2),
        nn.Flatten(),
        nn.Linear(128 * 3 * 3, 256), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())
