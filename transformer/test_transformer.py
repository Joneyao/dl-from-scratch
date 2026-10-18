import torch
from transformer.transformer import TransformerBlock, build_mini_transformer


def test_transformer_block_shape():
    block = TransformerBlock(d_model=32, n_heads=4)
    x = torch.randn(2, 6, 32)
    out = block(x)
    assert out.shape == x.shape


def test_mini_transformer_forward_shape():
    model = build_mini_transformer(d_model=32, n_heads=4, n_layers=2, vocab_size=50)
    x = torch.randint(0, 50, (2, 10))
    out = model(x)
    assert out.shape == (2, 10, 50)


def test_block_is_residual():
    # 残差连接让深层块的输出不会远离输入（范数不会爆炸）
    block = TransformerBlock(d_model=32, n_heads=4)
    x = torch.randn(4, 6, 32)
    out = block(x)
    assert out.norm() < x.norm() * 10
