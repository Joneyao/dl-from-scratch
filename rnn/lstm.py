"""#16 LSTM：手写门控拆解，对照 nn.LSTM。"""
import torch
import torch.nn as nn


def lstm_cell_forward(x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh):
    """手写 LSTM 单步：四组门（f/i/g/o）+ 细胞状态。

    gates = x·W_ih + h_prev·W_hh + b，切成四块：
      f=遗忘门、i=输入门、g=候选值、o=输出门
    """
    gates = x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh
    f, i, g, o = gates.chunk(4, dim=-1)
    f = torch.sigmoid(f)
    i = torch.sigmoid(i)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    c = f * c_prev + i * g          # 细胞状态：忘掉旧的 + 记住新的
    h = o * torch.tanh(c)           # 输出：门控决定"说多少"
    return h, c


def make_lstm(input_dim, hidden_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_ih = torch.randn(4 * hidden_dim, input_dim, generator=g) * 0.1
    W_hh = torch.randn(4 * hidden_dim, hidden_dim, generator=g) * 0.1
    b_ih = torch.zeros(4 * hidden_dim)
    b_hh = torch.zeros(4 * hidden_dim)
    return W_ih, W_hh, b_ih, b_hh


def lstm_forward(X, W_ih, W_hh, b_ih, b_hh):
    """对序列 X 逐步前向，返回所有时间步的 h 和最终的 c。"""
    batch = X.shape[1]
    hidden = W_hh.shape[1]
    h = torch.zeros(batch, hidden)
    c = torch.zeros(batch, hidden)
    outputs = []
    for t in range(X.shape[0]):
        h, c = lstm_cell_forward(X[t], h, c, W_ih, W_hh, b_ih, b_hh)
        outputs.append(h)
    return torch.stack(outputs), c
