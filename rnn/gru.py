"""#17 GRU：手写更简洁的门控 + 字符级文本生成。"""
import torch
import torch.nn as nn


def gru_cell_forward(x, h_prev, W_ir, W_iz, W_in, W_hr, W_hz, W_hn, b_ir, b_iz, b_in, b_hr, b_hz, b_hn):
    """手写 GRU 单步：重置门 r + 更新门 z，比 LSTM 少一个门。"""
    r = torch.sigmoid(x @ W_ir.T + b_ir + h_prev @ W_hr.T + b_hr)
    z = torch.sigmoid(x @ W_iz.T + b_iz + h_prev @ W_hz.T + b_hz)
    n = torch.tanh(x @ W_in.T + b_in + r * (h_prev @ W_hn.T + b_hn))
    h = (1 - z) * n + z * h_prev      # 更新门在"新旧状态"之间插值
    return h


class CharRNN(nn.Module):
    """字符级 RNN 语言模型：输入一个字符，预测下一个字符。"""

    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        emb = self.embed(x)                 # (batch, seq, hidden)
        out, h = self.rnn(emb, h)
        return self.fc(out), h


def train_char_rnn(text, seq_len=25, hidden_size=128, epochs=100, lr=0.01, seed=0):
    """训练一个字符级 RNN，返回 (model, char_to_idx, idx_to_char, loss_history)。"""
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    torch.manual_seed(seed)
    model = CharRNN(len(chars), hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []

    for _ in range(epochs):
        # 随机取一个片段作为输入，目标是它的下一个字符
        start = torch.randint(0, len(data) - seq_len - 1, (1,)).item()
        x = data[start:start + seq_len].unsqueeze(0)
        y = data[start + 1:start + seq_len + 1].unsqueeze(0)
        opt.zero_grad()
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, len(chars)), y.reshape(-1))
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return model, char_to_idx, idx_to_char, history


def generate(model, char_to_idx, idx_to_char, start_char, length=200, temperature=1.0):
    """用训练好的字符 RNN 续写文本。"""
    model.eval()
    chars = [start_char]
    idx = torch.tensor([[char_to_idx[start_char]]], dtype=torch.long)
    h = None
    with torch.no_grad():
        for _ in range(length):
            logits, h = model(idx, h)
            probs = torch.softmax(logits.squeeze() / temperature, dim=0)
            nxt = torch.multinomial(probs, 1).item()
            chars.append(idx_to_char[nxt])
            idx = torch.tensor([[nxt]], dtype=torch.long)
    return "".join(chars)
