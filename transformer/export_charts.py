"""运行 #18-#21 模块，导出真实数据配图到 charts_out/。"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from common.plotting import setup_cjk_font, save_fig
from transformer import attention, self_attention, bert_gpt

OUT = os.path.join(os.path.dirname(__file__), "charts_out")


def export_attention_heatmap():
    # query 一个词，看它对三个 key 的注意力权重分布
    _, attn = attention.demo_attention()
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(attn[0].detach().numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["词1", "词2", "词3"])
    ax.set_yticks([0])
    ax.set_yticklabels(["query"])
    plt.colorbar(im, ax=ax)
    ax.set_title("注意力权重：query 把重点放在哪些词上")
    for i in range(3):
        ax.text(i, 0, f"{attn[0,0,i].item():.2f}", ha="center", va="center", color="black")
    return save_fig(fig, OUT, "attention_heatmap.png")


def export_self_attention():
    torch.manual_seed(0)
    x = torch.randn(1, 6, 8)
    attn = self_attention.self_attention_weights(x).detach().numpy()[0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(attn, cmap="Blues", aspect="auto")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    labels = ["我", "爱", "深度", "学习", "很", "好"]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax)
    ax.set_title("自注意力：每个词对整句所有词的关注")
    return save_fig(fig, OUT, "self_attention.png")


def export_positional_encoding():
    pe = self_attention.positional_encoding(16, 8).numpy()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(pe.T, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    ax.set_xlabel("位置")
    ax.set_ylabel("维度")
    plt.colorbar(im, ax=ax)
    ax.set_title("正弦位置编码：每个位置有唯一指纹")
    return save_fig(fig, OUT, "positional_encoding.png")


def export_bert_gpt_mask():
    n = 6
    b = bert_gpt.bert_mask(n).numpy()
    g = bert_gpt.gpt_causal_mask(n).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, title in [(axes[0], b, "BERT 双向 mask（看整句）"),
                         (axes[1], g, "GPT 因果 mask（只看左边）")]:
        ax.imshow(m, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_title(title)
    return save_fig(fig, OUT, "bert_gpt_mask.png")


def main():
    setup_cjk_font()
    paths = [export_attention_heatmap(), export_self_attention(),
             export_positional_encoding(), export_bert_gpt_mask()]
    print("导出完成：")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
