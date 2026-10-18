import torch
from cnn.architectures import lenet, vgg_style, vgg_block, count_params


def test_lenet_forward_shape():
    model = lenet(num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    assert model(x).shape == (2, 10)


def test_vgg_style_forward_shape():
    model = vgg_style(num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    assert model(x).shape == (2, 10)


def test_vgg_block_halves_spatial():
    block = vgg_block(1, 8, num_convs=2)
    x = torch.randn(1, 1, 28, 28)
    out = block(x)
    assert out.shape == (1, 8, 14, 14)  # 池化后尺寸减半


def test_vgg_has_more_params_than_lenet():
    assert count_params(vgg_style()) > count_params(lenet())
