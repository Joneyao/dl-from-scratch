import numpy as np
from training.optimizer import make_quadratic, sgd_step, momentum_step, adam_step, optimize


def test_sgd_converges_on_quadratic():
    f, grad = make_quadratic(a=1.0, b=1.0)
    _, losses = optimize(sgd_step, f, grad, [3.0, 3.0], lr=0.1, n_iters=100)
    assert losses[-1] < losses[0] * 0.01


def test_adam_beats_sgd_on_ill_conditioned():
    # 病态函数 a≫b：SGD 在 b 方向收敛慢，Adam 自适应步长更快
    f, grad = make_quadratic(a=1.0, b=0.01)
    _, sgd_losses = optimize(sgd_step, f, grad, [3.0, 3.0], lr=0.1, n_iters=200)
    _, adam_losses = optimize(adam_step, f, grad, [3.0, 3.0], lr=0.1, n_iters=200)
    assert adam_losses[-1] < sgd_losses[-1]


def test_optimizers_trajectory_shape():
    f, grad = make_quadratic()
    traj, _ = optimize(sgd_step, f, grad, [2.0, 2.0], lr=0.1, n_iters=50)
    assert traj.shape == (51, 2)
