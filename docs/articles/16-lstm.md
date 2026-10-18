> 作者：轻松学AI
> 日期：2026-10-18
> 风格：通俗科普
> 状态：待发布

# 从零开始学深度学习 #16：LSTM——三条门与细胞状态

上一篇结尾留了个问题：RNN 记不住太远的信息，一句话开头的"张三"，到 50 个词之后就被忘了。

这一篇讲的 LSTM，就是为了解决这个问题设计的。它的思路很妙——给网络配一个"记忆筛选器"，什么该忘、什么该记，都由网络自己学着决定。

## 一、RNN 记不住长依赖，卡在哪

先想清楚 RNN 为什么记不远。它的记忆就是那个隐藏状态 `h`，每一步都被新的输入"冲刷"一遍：

```mermaid
---
title: RNN 的记忆每步被冲刷
---
flowchart LR
    A[记忆 h] --> B[乘权重 + 新输入]
    B --> C[旧的被冲淡]
    C --> D[远的信息丢了]
```

每一步，`h` 都被 `tanh` 压一次、被新输入混一次。几十步下来，最初的信息早就面目全非。要记住"张三"到第 50 个词，普通 RNN 力不从心。

## 二、LSTM：给记忆装三道门

LSTM 的破局，是引入一个独立的"细胞状态" `c`，专门负责长期记忆，再给它装三道门：

- **遗忘门**：决定丢掉哪些旧记忆。
- **输入门**：决定记住哪些新信息。
- **输出门**：决定当前输出多少。

```mermaid
---
title: LSTM 的细胞状态与三道门
---
flowchart LR
    A[旧记忆 c] --> B[遗忘门 决定丢多少]
    C[新信息] --> D[输入门 决定记多少]
    B --> E[新记忆 c_t]
    D --> E
    E --> F[输出门 决定说多少]
```

三道门的本质，都是 `sigmoid` 输出 0 到 1 之间的数，用来"开合"信息的流动。

![LSTM 三道门机制图](mechanism.png)
0 表示全关（彻底忘/不记/不说），1 表示全开。

![三道筛子门控插画](analogy.png)


## 三、手写 LSTM 的核心

手写 LSTM 的单步，就是把这三道门算出来。关键代码：

```python
gates = x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh
f, i, g, o = gates.chunk(4, dim=-1)      # 切出四块
f = torch.sigmoid(f)                     # 遗忘门
i = torch.sigmoid(i)                     # 输入门
g = torch.tanh(g)                        # 候选新值
o = torch.sigmoid(o)                     # 输出门
c = f * c_prev + i * g                   # 细胞状态：忘旧的 + 记新的
h = o * torch.tanh(c)                    # 输出：门控决定说多少
```

注意最后两行：细胞状态 `c` 的更新是"遗忘门乘旧记忆 + 输入门乘新信息"。这个结构让 LSTM 可以做到——某条重要信息，只要"遗忘门"一直开着、"输入门"不打扰，就能原样一路传下去，几十步都不衰减。

![LSTM 细胞状态随时间的演变](lstm_cell_state.png)

这张图是 LSTM 细胞状态在 10 个时间步里的变化。注意那些基本走平、甚至微微上升的曲线——信息被"保持"住了，而不是像 RNN 那样每步被冲刷。

## 四、LSTM 和 RNN 的本质区别

一句话总结：RNN 只有一个"记忆槽"，每步都整个重写；LSTM 有两个状态，其中细胞状态 `c` 走的是"加性"更新，不容易被冲淡。

```mermaid
---
title: RNN 一个槽 vs LSTM 两个状态
---
flowchart LR
    A[RNN 一个记忆槽] --> B[每步整个重写]
    C[LSTM 两个状态] --> D[细胞状态加性更新]
    D --> E[不易被冲淡]
```

这也解释了为什么 LSTM 在机器翻译、语音识别里统治了很多年——它终于让网络能记住"几十步之前的关键信息"。

下一篇看 LSTM 的简化版 GRU，以及用字符级 RNN 做一个真正的"文本生成"小实验。

- [colah 博客：Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [《动手学深度学习》9.2 长短期记忆网络（LSTM）](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)
- [《深度学习》花书 第 10 章](https://www.deeplearningbook.org/contents/rnn.html)
- [配套仓库 dl-from-scratch（rnn/lstm.py）](https://github.com/Joneyao/dl-from-scratch)
