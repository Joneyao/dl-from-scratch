"""#21 BERT vs GPT：双向 mask 与因果 mask 的区别。"""
import torch


def bert_mask(seq_len):
    """BERT 的 mask：全 1（双向，每个词能看整句）。"""
    return torch.ones(seq_len, seq_len)


def gpt_causal_mask(seq_len):
    """GPT 的因果 mask：下三角，每个词只能看它自己和左边的词。"""
    return torch.tril(torch.ones(seq_len, seq_len))


def apply_mask(scores, mask):
    """把 mask 为 0 的位置设成极小的数，softmax 后注意力趋近 0。"""
    return scores.masked_fill(mask == 0, -1e9)
