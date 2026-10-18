"""运行 #10-#14 模块，导出真实数据配图到 charts_out/。"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from common.plotting import setup_cjk_font, save_fig
from cnn import conv, pooling, architectures, resnet, practice

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def _letter_L():
    """合成一张 16x16 的 'L' 形灰度图。"""
    img = np.zeros((16, 16))
    img[2:14, 2] = 1.0
    img[12:14, 2:14] = 1.0
    return img


def export_conv_kernel():
    img = _letter_L()
    edges = conv.detect_edges(img)
    blurred = conv.blur(img)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, arr, title in [(axes[0], img, "原图"), (axes[1], edges, "边缘检测（拉普拉斯核）"),
                           (axes[2], blurred, "均值模糊核")]:
        ax.imshow(arr, cmap="Greys")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("同一个卷积操作，换个核就是完全不同的效果")
    return save_fig(fig, OUT, "conv_kernel.png")


def export_pooling():
    rng = np.random.default_rng(0)
    x = rng.random((16, 16))
    p1 = pooling.maxpool2d_naive(x, k=2, stride=2)
    p2 = pooling.maxpool2d_naive(p1, k=2, stride=2)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, arr, title in [(axes[0], x, "16×16"), (axes[1], p1, "池化后 8×8"),
                           (axes[2], p2, "再池化 4×4")]:
        ax.imshow(arr, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("最大池化：特征图越来越小，感受野越来越大")
    return save_fig(fig, OUT, "pooling.png")


def export_param_count():
    names = ["LeNet", "VGG 风格"]
    counts = [architectures.count_params(architectures.lenet()),
              architectures.count_params(architectures.vgg_style())]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, counts, color=["#2196F3", "#FF9800"])
    ax.set_ylabel("参数数量")
    ax.set_title("从 LeNet 到 VGG：参数暴涨，全靠堆深")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}", ha="center", va="bottom")
    return save_fig(fig, OUT, "param_count.png")


def _train_small(model, train_loader, epochs=2, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        for X, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
        losses.append(float(loss.item()))
    return losses


def export_resnet_vs_plain():
    ds = practice.get_fashion_mnist(root="cnn/data", train=True)
    sub = Subset(ds, range(1500))
    loader = DataLoader(sub, batch_size=64, shuffle=True)

    def plain_net():
        # 6 层普通卷积（无跳过连接）
        layers = [nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                  nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                  nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                  nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                  nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                  nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)]
        return nn.Sequential(*layers)

    torch.manual_seed(0)
    plain_loss = _train_small(plain_net(), loader, epochs=2)
    torch.manual_seed(0)
    res_loss = _train_small(resnet.build_resnet(num_classes=10, width=16), loader, epochs=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(plain_loss, "o-", label="普通 6 层网络", color="#EF5350")
    ax.plot(res_loss, "s-", label="6 层残差网络", color="#2196F3")
    ax.set_xlabel("epoch")
    ax.set_ylabel("损失")
    ax.set_title("同样深度：残差连接让训练更顺")
    ax.legend()
    return save_fig(fig, OUT, "resnet_vs_plain.png")


def export_cnn_training():
    train_ds = practice.get_fashion_mnist(root="cnn/data", train=True, augment=True)
    val_ds = practice.get_fashion_mnist(root="cnn/data", train=False)
    train_loader = DataLoader(Subset(train_ds, range(4000)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(val_ds, range(2000)), batch_size=128)
    torch.manual_seed(0)
    model = practice.small_cnn()
    hist = practice.train_cnn(model, train_loader, val_loader, epochs=4, lr=0.01)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["train_loss"], "o-", label="训练损失", color="#2196F3")
    axes[0].plot(hist["val_loss"], "s-", label="验证损失", color="#EF5350")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("损失")
    axes[0].set_title("Fashion-MNIST 训练/验证损失")
    axes[0].legend()
    axes[1].plot(hist["train_acc"], "o-", label="训练准确率", color="#2196F3")
    axes[1].plot(hist["val_acc"], "s-", label="验证准确率", color="#EF5350")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("训练/验证准确率")
    axes[1].legend()
    return save_fig(fig, OUT, "cnn_training.png")


def main():
    setup_cjk_font()
    paths = [export_conv_kernel(), export_pooling(), export_param_count(),
             export_resnet_vs_plain(), export_cnn_training()]
    print("导出完成：")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
