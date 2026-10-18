"""运行 #01/#03 模块与 #02 历史数据，导出真实数据配图到 charts_out/。"""
import os
import numpy as np
import matplotlib.pyplot as plt
from common.plotting import setup_cjk_font, save_fig
from common.datasets import make_spirals, make_linear_data
from intro.deep_vs_shallow import shallow_net, deep_net, train, accuracy, decision_grid
from intro.pytorch_basics import linear_regression_torch

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_decision_boundary():
    X, y = make_spirals(n=400, noise=0.1, seed=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, net_fn, title in [
        (axes[0], shallow_net, "浅层 MLP（1 个隐藏层）"),
        (axes[1], deep_net, "深层 MLP（3 个隐藏层）"),
    ]:
        model = net_fn()
        train(model, X, y, n_iters=1200, seed=0)
        gx, gy, z = decision_grid(model)
        ax.contourf(gx, gy, z, levels=30, cmap="coolwarm", alpha=.7)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#2196F3", s=8, label="螺旋一")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#FFCA28", s=8, label="螺旋二")
        acc = accuracy(model, X, y)
        ax.set_title(f"{title}｜准确率 {acc:.0%}")
        ax.legend(fontsize=9)
    fig.suptitle("同宽 8 神经元：深度让网络能分开两条螺旋")
    return save_fig(fig, OUT, "decision_boundary.png")


def export_loss_curves():
    X, y = make_spirals(n=400, noise=0.1, seed=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for net_fn, label in [(shallow_net, "浅层"), (deep_net, "深层")]:
        hist = train(net_fn(), X, y, n_iters=1200, seed=0)
        ax.plot(hist, label=label)
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("损失")
    ax.set_title("浅层 vs 深层：损失下降对比")
    ax.legend()
    ax.set_yscale("log")
    return save_fig(fig, OUT, "train_loss_curve.png")


def export_imagenet_timeline():
    # ImageNet 挑战赛 Top-5 错误率，数据来自 AlexNet 论文与公开史实（写 #02 时再次核实）
    years = [2010, 2011, 2012, 2013, 2014, 2015]
    top5_err = [28.2, 25.8, 16.4, 11.7, 7.3, 3.57]
    labels = ["2010\n手工特征", "2011\n手工特征", "2012\nAlexNet", "2013", "2014\nGoogLeNet", "2015\nResNet"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, top5_err, "o-", color="#2196F3", linewidth=2.5)
    for x, y, lab in zip(years, top5_err, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8)
    ax.set_xlabel("年份")
    ax.set_ylabel("Top-5 错误率 (%)")
    ax.set_title("ImageNet 挑战赛：深度学习登场的 2012")
    ax.grid(alpha=.3)
    return save_fig(fig, OUT, "imagenet_timeline.png")


def export_linear_fit_torch():
    X, y = make_linear_data(n=60, slope=2.0, intercept=1.0, noise=1.2, seed=3)
    w, b, history = linear_regression_torch(X, y, lr=0.05, n_iters=500)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(X, y, c="#2196F3", s=18, label="数据")
    xs = np.linspace(X.min(), X.max(), 100)
    axes[0].plot(xs, w * xs + b, color="#FFCA28", linewidth=2.5,
                 label=f"拟合直线 y={w:.2f}x+{b:.2f}")
    axes[0].set_title("autograd 手动梯度下降拟合直线")
    axes[0].legend()
    axes[1].plot(history, color="#2196F3")
    axes[1].set_xlabel("迭代轮数")
    axes[1].set_ylabel("损失")
    axes[1].set_title("损失下降过程")
    axes[1].set_yscale("log")
    return save_fig(fig, OUT, "linear_fit_torch.png")


def export_autograd_tangent():
    # y = x^2 + 2x + 1，在 x=3 处的切线斜率 dy/dx = 2x+2 = 8
    x = np.linspace(-1, 4, 200)
    y = x ** 2 + 2 * x + 1
    x0 = 3.0
    y0 = x0 ** 2 + 2 * x0 + 1
    slope = 2 * x0 + 2
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, color="#2196F3", linewidth=2.5, label="y = x² + 2x + 1")
    ax.plot(x, slope * (x - x0) + y0, color="#FFCA28", linewidth=2,
            linestyle="--", label=f"x=3 处切线（斜率 {slope}）")
    ax.scatter([x0], [y0], color="#EF5350", s=60, zorder=5)
    ax.annotate(f"斜率 = 2x+2 = {slope}", (x0, y0), textcoords="offset points",
                xytext=(15, 15), fontsize=11, color="#EF5350")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("autograd 求的导，就是切线的斜率")
    ax.legend()
    ax.grid(alpha=.3)
    return save_fig(fig, OUT, "autograd_tangent.png")


def main():
    setup_cjk_font()
    paths = [
        export_decision_boundary(),
        export_loss_curves(),
        export_imagenet_timeline(),
        export_linear_fit_torch(),
        export_autograd_tangent(),
    ]
    print("导出完成：")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
