import numpy as np
from training import activation as act


def test_sigmoid_center_and_bounds():
    assert abs(act.sigmoid(0.0) - 0.5) < 1e-9
    assert act.sigmoid(100) > 0.999
    assert act.sigmoid(-100) < 0.001


def test_sigmoid_deriv_max_is_quarter():
    # sigmoid 导数最大 0.25，这是梯度消失的根因
    x = np.linspace(-10, 10, 1000)
    assert np.max(act.sigmoid_deriv(x)) <= 0.25 + 1e-9


def test_relu_and_deriv():
    assert act.relu(-3.0) == 0.0
    assert act.relu(3.0) == 3.0
    assert act.relu_deriv(-1.0) == 0.0
    assert act.relu_deriv(1.0) == 1.0


def test_tanh_deriv():
    assert abs(act.tanh(0.0)) < 1e-9
    assert abs(act.tanh_deriv(0.0) - 1.0) < 1e-9


def test_gelu_near_zero():
    assert abs(act.gelu(0.0)) < 1e-6
    assert act.gelu(5.0) > 4.9  # GELU 在正数侧接近恒等
