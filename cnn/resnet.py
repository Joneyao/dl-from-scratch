"""#13 残差连接：ResNet 的 BasicBlock（跳过连接）。"""
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet 的基本残差块：两个 3x3 卷积 + 跳过连接。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 当通道数或尺寸变化时，跳过连接要经过 1x1 卷积对齐
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity          # 关键：加上跳过来的原始输入
        return self.relu(out)


def build_resnet(num_classes=10, width=32):
    """一个小型 ResNet：三组残差块。"""
    return nn.Sequential(
        nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
        BasicBlock(width, width),
        BasicBlock(width, width * 2, stride=2),
        BasicBlock(width * 2, width * 2),
        BasicBlock(width * 2, width * 4, stride=2),
        BasicBlock(width * 4, width * 4),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(width * 4, num_classes),
    )
