import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    """LSTM Encoder：把输入句子编码成 hidden/cell 语义状态。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, pad_idx=0):
        """定义 embedding 层和 LSTM 层。"""
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, src_ids, src_lengths):
        """输入源句子 id 和真实长度，输出最后的 hidden/cell。

        输入:
            src_ids: [batch_size, seq_len]
            src_lengths: [batch_size]
        中间:
            embedded: [batch_size, seq_len, embed_dim]
        输出:
            hidden: [num_layers, batch_size, hidden_dim]
            cell: [num_layers, batch_size, hidden_dim]
        """
        embedded = self.embedding(src_ids)
        packed_embedded = pack_padded_sequence(
            input=embedded,
            lengths=src_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, cell) = self.lstm(packed_embedded)
        return hidden, cell


class Decoder(nn.Module):
    """LSTM Decoder：根据 Encoder 状态逐步预测目标 token。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, pad_idx=0):
        """定义 embedding、LSTM 和输出到词表大小的线性层。"""
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_input_ids, hidden, cell):
        """输入 decoder 当前 token 序列和状态，输出每个位置的词表 logits。

        输入:
            tgt_input_ids: [batch_size, tgt_len]
            hidden/cell: [num_layers, batch_size, hidden_dim]
        中间:
            embeded: [batch_size, tgt_len, embed_dim]
            outputs: [batch_size, tgt_len, hidden_dim]
        输出:
            logits: [batch_size, tgt_len, vocab_size]
            hidden/cell: [num_layers, batch_size, hidden_dim]
        """
        embeded = self.embedding(tgt_input_ids)
        outputs, (hidden, cell) = self.lstm(embeded, (hidden, cell))
        logits = self.fc(outputs)
        return logits, (hidden, cell)


class Seq2Seq(nn.Module):
    """完整自编码模型：Encoder 读原句，Decoder 重构原句。"""

    def __init__(self, encoder, decoder):
        """把已经定义好的 encoder 和 decoder 组合起来。"""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_ids, src_lengths):
        """训练时使用 teacher forcing，返回 logits 和右移后的 target。

        输入:
            src_ids: [batch_size, seq_len]
            src_lengths: [batch_size]
        切片后:
            decoder_input: [batch_size, seq_len - 1]
            decoder_target: [batch_size, seq_len - 1]
        输出:
            logits: [batch_size, seq_len - 1, vocab_size]
            decoder_target: [batch_size, seq_len - 1]
        """
        hidden, cell = self.encoder(src_ids, src_lengths)
        decoder_input = src_ids[:, :-1]
        decoder_target = src_ids[:, 1:]
        logits, _ = self.decoder(decoder_input, hidden, cell)
        return logits, decoder_target
