import torch
from transformer.attention import scaled_dot_product_attention, attention_weights, demo_attention


def test_attention_output_shape():
    torch.manual_seed(0)
    Q = torch.randn(2, 3, 8)
    K = torch.randn(2, 5, 8)
    V = torch.randn(2, 5, 8)
    out, attn = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 3, 8)
    assert attn.shape == (2, 3, 5)


def test_attention_weights_sum_to_one():
    torch.manual_seed(0)
    Q = torch.randn(1, 3, 8)
    K = torch.randn(1, 5, 8)
    attn = attention_weights(Q, K)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 3), atol=1e-5)


def test_attention_mask_focuses_elsewhere():
    torch.manual_seed(0)
    Q = torch.randn(1, 1, 8)
    K = torch.randn(1, 3, 8)
    V = torch.randn(1, 3, 8)
    # 把第一个位置 mask 掉，注意力应该集中在剩余位置
    mask = torch.tensor([[[0, 1, 1]]])
    out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert attn[0, 0, 0].item() < 1e-4


def test_demo_runs():
    out, attn = demo_attention()
    assert out.shape == (1, 1, 4)
