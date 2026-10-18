import torch
from cnn.resnet import BasicBlock, build_resnet


def test_basicblock_identity_shape():
    block = BasicBlock(8, 8)
    x = torch.randn(2, 8, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_basicblock_downsample_shape():
    block = BasicBlock(8, 16, stride=2)
    x = torch.randn(2, 8, 16, 16)
    out = block(x)
    assert out.shape == (2, 16, 8, 8)


def test_resnet_forward_shape():
    model = build_resnet(num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    assert model(x).shape == (2, 10)


def test_resnet_has_multiple_blocks():
    model = build_resnet()
    assert sum(1 for m in model if isinstance(m, BasicBlock)) == 5
