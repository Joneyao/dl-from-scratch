"""#20 Transformer 架构：手搭一个 mini 编码器-解码器。"""
import torch
import torch.nn as nn
import math


class TransformerBlock(nn.Module):
    """一个 Transformer 块：多头自注意力 + 前馈网络，每处都带残差 + LayerNorm。"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)       # 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)        # 残差 + LayerNorm
        return x


def build_mini_transformer(d_model=64, n_heads=4, n_layers=2, vocab_size=100):
    """一个 mini Transformer：embedding + 若干 TransformerBlock + 输出头。"""
    return nn.Sequential(
        nn.Embedding(vocab_size, d_model),
        *[TransformerBlock(d_model, n_heads) for _ in range(n_layers)],
        nn.Linear(d_model, vocab_size),
    )


def sinusoidal_position_encoding(seq_len, d_model):
    from transformer.self_attention import positional_encoding
    return positional_encoding(seq_len, d_model)
