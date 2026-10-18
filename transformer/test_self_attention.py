import torch
from transformer.self_attention import positional_encoding, MultiHeadSelfAttention, self_attention_weights


def test_positional_encoding_shape_and_values():
    pe = positional_encoding(10, 8)
    assert pe.shape == (10, 8)
    # 位置编码是正弦/余弦，值在 [-1, 1]
    assert pe.abs().max() <= 1.0


def test_positional_encoding_distinguishes_positions():
    pe = positional_encoding(5, 8)
    # 不同位置编码不同
    assert not torch.allclose(pe[0], pe[1])


def test_multihead_forward_shape():
    mha = MultiHeadSelfAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 5, 16)
    out = mha(x)
    assert out.shape == x.shape


def test_self_attention_weights_sum_to_one():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    attn = self_attention_weights(x)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 4), atol=1e-5)
