"""运行 #04-#09 模块，导出真实数据配图到 charts_out/。"""
import os
import numpy as np
import matplotlib.pyplot as plt
from common.plotting import setup_cjk_font, save_fig
from training import gradient_flow, activation as act, init, normalization, optimizer, regularization

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_gradient_flow():
    sig, relu = gradient_flow.compare_sigmoid_vs_relu(depth=20, seed=0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sig, "o-", color="#EF5350", label="Sigmoid 网络")
    ax.plot(relu, "s-", color="#2196F3", label="ReLU 网络")
    ax.set_yscale("log")
    ax.set_xlabel("层编号（0 = 最靠近输入）")
    ax.set_ylabel("该层梯度范数（对数）")
    ax.set_title("梯度消失：Sigmoid 深层网络的前几层几乎学不到东西")
    ax.legend()
    ax.grid(alpha=.3)
    return save_fig(fig, OUT, "gradient_flow.png")


def export_activations():
    x = np.linspace(-5, 5, 500)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, fn, color in [("Sigmoid", act.sigmoid, "#EF5350"), ("Tanh", act.tanh, "#8E24AA"),
                            ("ReLU", act.relu, "#2196F3"), ("GELU", act.gelu, "#FF9800")]:
        axes[0].plot(x, fn(x), label=name, color=color, linewidth=2)
    axes[0].set_title("各激活函数的形状")
    axes[0].legend()
    for name, fn, color in [("Sigmoid", act.sigmoid_deriv, "#EF5350"), ("Tanh", act.tanh_deriv, "#8E24AA"),
                            ("ReLU", act.relu_deriv, "#2196F3")]:
        axes[1].plot(x, fn(x), label=name, color=color, linewidth=2)
    axes[1].set_title("导数：Sigmoid 最大只有 0.25，ReLU 恒为 1")
    axes[1].legend()
    axes[1].axhline(0.25, color="#EF5350", linestyle=":", alpha=.6)
    return save_fig(fig, OUT, "activations.png")


def export_init():
    result = init.compare_init(depth=20, seed=0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, stds in result.items():
        ax.plot(stds, "o-", label=name, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("层编号")
    ax.set_ylabel("激活标准差（对数）")
    ax.set_title("权重初始化决定信号是衰减还是爆炸")
    ax.legend()
    ax.grid(alpha=.3)
    return save_fig(fig, OUT, "init_comparison.png")


def export_batchnorm():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 4)).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1]).reshape(-1, 1).astype(np.float32)
    hist_bn = normalization.train_torch_with_without_bn(X, y, use_bn=True, n_iters=800, seed=0)
    hist_no = normalization.train_torch_with_without_bn(X, y, use_bn=False, n_iters=800, seed=0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(hist_bn, label="加 BatchNorm", color="#2196F3", linewidth=2)
    ax.plot(hist_no, label="不加", color="#EF5350", linewidth=2)
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("损失")
    ax.set_title("BatchNorm 让训练更稳、收敛更快")
    ax.legend()
    ax.set_yscale("log")
    return save_fig(fig, OUT, "batchnorm_effect.png")


def export_optimizers():
    f, grad = optimizer.make_quadratic(a=1.0, b=0.02)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, step_fn in [("SGD", optimizer.sgd_step), ("动量", optimizer.momentum_step),
                          ("Adam", optimizer.adam_step)]:
        traj, losses = optimizer.optimize(step_fn, f, grad, [3.0, 3.0], lr=0.1, n_iters=200)
        axes[0].plot(traj[:, 0], traj[:, 1], "o-", label=name, markersize=3, linewidth=1.5)
        axes[1].plot(losses, label=name, linewidth=2)
    axes[0].set_title("病态二次函数上的优化轨迹")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[1].set_title("损失下降对比")
    axes[1].set_yscale("log")
    axes[1].legend()
    return save_fig(fig, OUT, "optimizers.png")


def export_regularization():
    degrees = list(range(1, 15))
    tr_list, te_list = [], []
    for d in degrees:
        tr, te = regularization.train_test_errors(degree=d, alpha=0.0, seed=0)
        tr_list.append(tr)
        te_list.append(te)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(degrees, tr_list, "o-", label="训练误差", color="#2196F3")
    ax.plot(degrees, te_list, "s-", label="测试误差", color="#EF5350")
    ax.set_xlabel("多项式阶数")
    ax.set_ylabel("均方误差")
    ax.set_title("过拟合：阶数升高后测试误差抬头")
    ax.legend()
    ax.set_yscale("log")
    return save_fig(fig, OUT, "overfitting_curve.png")


def main():
    setup_cjk_font()
    paths = [export_gradient_flow(), export_activations(), export_init(),
             export_batchnorm(), export_optimizers(), export_regularization()]
    print("导出完成：")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
