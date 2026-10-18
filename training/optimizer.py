"""#08 优化器：手写 SGD / 动量 / Adam 的更新规则与收敛对比。"""
import numpy as np


def make_quadratic(a=1.0, b=0.1):
    """一个病态的二维二次函数 f = a·x² + b·y²，a≫b 时梯度下降会震荡。"""
    def f(xy):
        x, y = xy
        return a * x ** 2 + b * y ** 2

    def grad(xy):
        x, y = xy
        return np.array([2 * a * x, 2 * b * y])
    return f, grad


def sgd_step(params, grads, lr, state):
    return params - lr * grads, state


def momentum_step(params, grads, lr, state, momentum=0.9):
    v = momentum * state.get("v", np.zeros_like(params)) + grads
    return params - lr * v, {"v": v}


def adam_step(params, grads, lr, state, t, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * state.get("m", np.zeros_like(params)) + (1 - beta1) * grads
    v = beta2 * state.get("v", np.zeros_like(params)) + (1 - beta2) * grads ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return params - lr * m_hat / (np.sqrt(v_hat) + eps), {"m": m, "v": v}


def optimize(step_fn, f, grad, x0, lr, n_iters):
    """跑一步法优化器，返回每轮的参数轨迹和损失。"""
    params = np.array(x0, dtype=float)
    state = {}
    trajectory = [params.copy()]
    losses = [f(params)]
    for t in range(1, n_iters + 1):
        g = grad(params)
        if step_fn is adam_step:
            params, state = step_fn(params, g, lr, state, t)
        else:
            params, state = step_fn(params, g, lr, state)
        trajectory.append(params.copy())
        losses.append(f(params))
    return np.array(trajectory), losses
