import torch
from rnn.gru import gru_cell_forward, train_char_rnn, generate


def _rand_gru(hidden=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_dim = 4
    W = lambda a, b: torch.randn(a, b, generator=g) * 0.1
    b = lambda a: torch.zeros(a)
    return (input_dim, W(hidden, input_dim), W(hidden, input_dim), W(hidden, input_dim),
            W(hidden, hidden), W(hidden, hidden), W(hidden, hidden),
            b(hidden), b(hidden), b(hidden), b(hidden), b(hidden), b(hidden))


def test_gru_cell_shape():
    input_dim, W_ir, W_iz, W_in, W_hr, W_hz, W_hn, b_ir, b_iz, b_in, b_hr, b_hz, b_hn = _rand_gru()
    x = torch.randn(2, input_dim)
    h = torch.randn(2, 8)
    out = gru_cell_forward(x, h, W_ir, W_iz, W_in, W_hr, W_hz, W_hn, b_ir, b_iz, b_in, b_hr, b_hz, b_hn)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_char_rnn_reduces_loss():
    text = "hello world hello world hello world "
    model, c2i, i2c, hist = train_char_rnn(text, seq_len=10, hidden_size=32, epochs=150, lr=0.01, seed=0)
    assert hist[-1] < hist[0]


def test_generate_returns_text():
    text = "abcabcabcabcabcabcabcabc"
    model, c2i, i2c, hist = train_char_rnn(text, seq_len=5, hidden_size=32, epochs=100, lr=0.01, seed=0)
    out = generate(model, c2i, i2c, "a", length=30)
    assert isinstance(out, str) and len(out) == 31
