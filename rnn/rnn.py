"""#15 RNN：手写单步循环前向，对照 nn.RNN。"""
import torch
import torch.nn as nn


def rnn_cell_forward(x, h_prev, W_xh, W_hh, b_h):
    """RNN 单步：h_t = tanh(x_t·W_xh + h_{t-1}·W_hh + b_h)。"""
    return torch.tanh(x @ W_xh + h_prev @ W_hh + b_h)


def rnn_forward(X, h0, W_xh, W_hh, b_h):
    """对序列 X（shape: seq_len, batch, input_dim）逐步前向，返回所有时间步的 h。

    注意：同一个 W_xh/W_hh 在所有时间步反复使用——这就是"记忆"的代价和来源。
    """
    h = h0
    outputs = []
    for t in range(X.shape[0]):
        h = rnn_cell_forward(X[t], h, W_xh, W_hh, b_h)
        outputs.append(h)
    return torch.stack(outputs)


def make_rnn(input_dim, hidden_dim, seed=0):
    """随机初始化一个 RNN 的权重，返回 (W_xh, W_hh, b_h)。"""
    g = torch.Generator().manual_seed(seed)
    W_xh = torch.randn(input_dim, hidden_dim, generator=g) * 0.1
    W_hh = torch.randn(hidden_dim, hidden_dim, generator=g) * 0.1
    b_h = torch.zeros(hidden_dim)
    return W_xh, W_hh, b_h


def char_to_tensor(text, char_to_idx):
    return torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)


def make_char_maps(text):
    """建 char<->idx 映射。"""
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return chars, char_to_idx, idx_to_char
