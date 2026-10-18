# 参考资料

本仓库的代码根据官方教程、经典教材和开源教学仓库的原理**从零重新实现**，只借鉴思路和边界处理，不摘抄任何开源项目源码。每个技术点先手写最小实现，再用 PyTorch 框架版做数值对照验证正确性。

## 官方教程

| 教程 | 用途 |
|------|------|
| [PyTorch Tutorials](https://pytorch.org/tutorials/)（60-Minute Blitz / Learn the Basics / Transfer Learning / NLP from Scratch） | 框架 API 与训练循环的官方基准，导论与工具篇（#03）、实战篇（#14）核心依据 |

## 经典书籍（原理讲解的深度锚点）

| 书 | 一句话定位 | 主要参考章节 |
|----|-----------|-------------|
| 《动手学深度学习》d2l（李沐等） | 中文读者最流行的深度学习入门，PyTorch 版 | 各篇对应章：卷积 / 现代 CNN / RNN / 注意力 / 优化算法 |
| 《深度学习》花书（Goodfellow et al.） | 深度学习理论圣经 | Ch1 导论、Ch6-8 训练与优化、Ch9-10 CNN / RNN |
| 邱锡鹏《神经网络与深度学习》 | 中文经典教材，公式严谨 | CNN / RNN / 注意力各章 |
| 斋藤康毅《深度学习入门》（鱼书） | 手写实现风格标杆 | CNN 卷积 / 池化手写 |
| fastai《Practical Deep Learning for Coders》 | 自上而下实战思路 | 图像分类实战 |

## 开源教学仓库（代码实现的对照 / 灵感）

**原则：只借鉴思路和边界处理，代码从原理重新实现，不复制源码。**

| 项目 | 用途 |
|------|------|
| [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh) | 动手学深度学习官方实现，章节结构与代码对照 |
| [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | 手写反向传播与神经网络的标杆 |
| [oreilly-japan/deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch) | 鱼书官方实现，CNN 手写风格参考 |
| [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) | Transformer 逐行注释，Transformer 篇参考 |
| [fastai/fastbook](https://github.com/fastai/fastbook) | fastai 课程配套书，实战思路参考 |

## 数据集

| 数据集 | 来源 | 用在哪 |
|--------|------|--------|
| 合成数据 | `common/`（带噪声回归/分类、损失曲面） | 导论与训练地基篇 |
| Fashion-MNIST / CIFAR-10 | torchvision 自动下载 | CNN 篇 |
| 字符级文本（公开语料） | 公开文本 | RNN 序列生成篇 |
| toy 序列 / 翻译数据 | `common/` 合成 | 注意力与 Transformer 篇 |
