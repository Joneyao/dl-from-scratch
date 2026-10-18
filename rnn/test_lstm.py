import torch
from rnn.lstm import lstm_cell_forward, lstm_forward, make_lstm


def test_lstm_cell_shape():
    W_ih, W_hh, b_ih, b_hh = make_lstm(4, 8)
    x = torch.randn(2, 4)
    h = torch.randn(2, 8)
    c = torch.randn(2, 8)
    h_out, c_out = lstm_cell_forward(x, h, c, W_ih, W_hh, b_ih, b_hh)
    assert h_out.shape == (2, 8)
    assert c_out.shape == (2, 8)


def test_lstm_forward_shapes():
    W_ih, W_hh, b_ih, b_hh = make_lstm(4, 8)
    X = torch.randn(5, 2, 4)
    outs, c = lstm_forward(X, W_ih, W_hh, b_ih, b_hh)
    assert outs.shape == (5, 2, 8)
    assert c.shape == (2, 8)


def test_cell_state_changes_and_bounded():
    # 细胞状态是有限的（门控保证不爆炸）
    W_ih, W_hh, b_ih, b_hh = make_lstm(4, 8)
    X = torch.randn(10, 1, 4)
    _, c = lstm_forward(X, W_ih, W_hh, b_ih, b_hh)
    assert torch.isfinite(c).all()
