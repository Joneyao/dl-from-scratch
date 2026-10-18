"""#19 自注意力 + 多头注意力 + 位置编码。"""
import torch
import torch.nn as nn
import math


def positional_encoding(seq_len, d_model):
    """正弦位置编码：给每个位置一个唯一的向量，让模型知道"顺序"。"""
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MultiHeadSelfAttention(nn.Module):
    """手写多头自注意力：Q/K/V 都来自同一个序列。"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.Wq(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)
        return self.Wo(out)


def self_attention_weights(x, d_model=8):
    """可视化一个序列里"每个词对其他所有词的注意力分布"。"""
    Wq = nn.Linear(d_model, d_model)
    Wk = nn.Linear(d_model, d_model)
    Q = Wq(x)
    K = Wk(x)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_model)
    return torch.softmax(scores, dim=-1)
