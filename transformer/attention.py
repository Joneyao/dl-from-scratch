"""#18 注意力机制：手写缩放点积注意力，对照 nn.MultiheadAttention。"""
import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """注意力 = softmax(Q·Kᵀ / √d_k) · V。返回 (输出, 注意力权重)。"""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=-1)
    return attn @ V, attn


def attention_weights(Q, K):
    """只算注意力权重（用于可视化"这个词在关注哪些词"）。"""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    return torch.softmax(scores, dim=-1)


def demo_attention():
    """一个玩具演示：query 查询，key/value 是三个词，看 attention 权重如何聚焦。"""
    torch.manual_seed(0)
    d = 4
    Q = torch.randn(1, 1, d)
    K = torch.randn(1, 3, d)
    V = torch.randn(1, 3, d)
    out, attn = scaled_dot_product_attention(Q, K, V)
    return out, attn
