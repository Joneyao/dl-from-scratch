import torch
import torch.nn as nn
from rnn.rnn import rnn_cell_forward, rnn_forward, make_rnn, make_char_maps


def test_rnn_cell_shape():
    W_xh, W_hh, b_h = make_rnn(4, 8)
    x = torch.randn(2, 4)
    h = torch.randn(2, 8)
    out = rnn_cell_forward(x, h, W_xh, W_hh, b_h)
    assert out.shape == (2, 8)


def test_rnn_forward_matches_torch():
    torch.manual_seed(0)
    input_dim, hidden_dim = 4, 8
    W_xh, W_hh, b_h = make_rnn(input_dim, hidden_dim, seed=0)
    X = torch.randn(5, 2, input_dim)  # seq=5, batch=2
    h0 = torch.zeros(2, hidden_dim)
    mine = rnn_forward(X, h0, W_xh, W_hh, b_h)

    rnn = nn.RNN(input_dim, hidden_dim, bias=True)
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(W_xh.T)
        rnn.weight_hh_l0.copy_(W_hh.T)
        rnn.bias_ih_l0.copy_(b_h)
        rnn.bias_hh_l0.copy_(torch.zeros(hidden_dim))
    theirs, _ = rnn(X)
    assert torch.allclose(mine, theirs, atol=1e-5)


def test_rnn_reuses_same_weights():
    # RNN 只有一个网络，所有时间步共用同一组权重——这就是"记忆"的来源
    W_xh, W_hh, b_h = make_rnn(4, 8)
    X = torch.randn(5, 2, 4)
    out = rnn_forward(X, torch.zeros(2, 8), W_xh, W_hh, b_h)
    assert out.shape == (5, 2, 8)


def test_char_maps_roundtrip():
    text = "hello world"
    chars, c2i, i2c = make_char_maps(text)
    assert len(chars) == len(set(text))
    assert i2c[c2i["h"]] == "h"
