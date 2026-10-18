import torch
from transformer.bert_gpt import bert_mask, gpt_causal_mask, apply_mask


def test_bert_mask_all_ones():
    m = bert_mask(5)
    assert torch.all(m == 1)


def test_gpt_causal_mask_lower_triangular():
    m = gpt_causal_mask(4)
    expected = torch.tensor([[1, 0, 0, 0],
                             [1, 1, 0, 0],
                             [1, 1, 1, 0],
                             [1, 1, 1, 1]], dtype=torch.float)
    assert torch.allclose(m, expected)


def test_gpt_mask_blocks_future():
    # GPT 每个词只能看自己和左边，不能看右边（未来）
    m = gpt_causal_mask(3)
    assert m[0, 1] == 0 and m[0, 2] == 0
    assert m[2, 0] == 1 and m[2, 2] == 1


def test_apply_mask():
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.float)
    masked = apply_mask(scores, mask)
    assert masked[0, 2] < -1e8
    assert masked[0, 0] == 1.0
