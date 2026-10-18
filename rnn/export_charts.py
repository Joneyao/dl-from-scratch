"""运行 #15-#17 模块，导出真实数据配图到 charts_out/。"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from common.plotting import setup_cjk_font, save_fig
from rnn import rnn, lstm, gru

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_char_loss():
    text = "hello world hello world hello world hello world "
    model, c2i, i2c, hist = gru.train_char_rnn(text, seq_len=10, hidden_size=64,
                                               epochs=200, lr=0.01, seed=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist, color="#2196F3")
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("损失")
    ax.set_title("字符级 RNN 训练：损失一路下降")
    ax.set_yscale("log")
    return save_fig(fig, OUT, "rnn_loss.png")


def export_lstm_cell_state():
    # 喂一个序列，看 LSTM 的细胞状态如何逐步"记住"信息
    W_ih, W_hh, b_ih, b_hh = lstm.make_lstm(4, 8, seed=0)
    X = torch.randn(10, 1, 4)
    h = torch.zeros(1, 8)
    c = torch.zeros(1, 8)
    cs = []
    for t in range(10):
        h, c = lstm.lstm_cell_forward(X[t], h, c, W_ih, W_hh, b_ih, b_hh)
        cs.append(c[0].detach().numpy())
    cs = np.array(cs)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cs)
    ax.set_xlabel("时间步")
    ax.set_ylabel("细胞状态值")
    ax.set_title("LSTM 细胞状态随时间的演变")
    return save_fig(fig, OUT, "lstm_cell_state.png")


def export_gru_generation():
    text = "abcabcabcabcabcabcabcabcabcabcabc"
    model, c2i, i2c, hist = gru.train_char_rnn(text, seq_len=8, hidden_size=32,
                                               epochs=150, lr=0.01, seed=0)
    generated = gru.generate(model, c2i, i2c, "a", length=60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist, color="#2196F3")
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("损失")
    ax.set_title(f"GRU 训练后续写：{generated[:40]}...")
    ax.set_yscale("log")
    # 把生成的文本存一份，供文章引用
    with open(os.path.join(OUT, "generated_text.txt"), "w") as f:
        f.write(generated)
    return save_fig(fig, OUT, "gru_loss.png")


def main():
    setup_cjk_font()
    paths = [export_char_loss(), export_lstm_cell_state(), export_gru_generation()]
    print("导出完成：")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
