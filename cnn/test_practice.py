import torch
from cnn.practice import small_cnn, train_cnn


def test_small_cnn_forward_shape():
    model = small_cnn()
    x = torch.randn(2, 1, 28, 28)
    assert model(x).shape == (2, 10)


def test_train_cnn_reduces_loss():
    torch.manual_seed(0)
    model = small_cnn(num_classes=2)
    # 构造清晰的空域信号：左上角亮→类别0，右下角亮→类别1
    X = torch.rand(128, 1, 28, 28) * 0.1
    y = torch.zeros(128, dtype=torch.long)
    for i in range(128):
        if i % 2 == 0:
            X[i, 0, :14, :14] = 1.0
            y[i] = 0
        else:
            X[i, 0, 14:, 14:] = 1.0
            y[i] = 1
    loader = [(X, y)]
    hist = train_cnn(model, loader, loader, epochs=3, lr=0.01)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
