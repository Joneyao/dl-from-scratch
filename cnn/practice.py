"""#14 CNN 实战：Fashion-MNIST 训练循环 + 数据增强。"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def get_fashion_mnist(root="cnn/data", train=True, augment=False):
    """加载 Fashion-MNIST（自动下载）。augment=True 时加数据增强。"""
    if augment:
        t = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        t = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.FashionMNIST(root=root, train=train, download=True, transform=t)


def small_cnn(num_classes=10):
    """一个小型 CNN：两个卷积块 + 全连接头。"""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, num_classes),
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device="cpu"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn, device="cpu"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = loss_fn(out, y)
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


def train_cnn(model, train_loader, val_loader, epochs=3, lr=0.01, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        tl, ta = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        vl, va = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
    return history
