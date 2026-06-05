import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, pad_idx=0):
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
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, pad_idx=0):
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
        embeded = self.embedding(tgt_input_ids)
        outputs, (hidden, cell) = self.lstm(embeded, (hidden, cell))
        logits = self.fc(outputs)
        return logits, (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_ids, src_lengths):
        hidden, cell = self.encoder(src_ids, src_lengths)
        decoder_input = src_ids[:, :-1]
        decoder_target = src_ids[:, 1:]
        logits, _ = self.decoder(decoder_input, hidden, cell)
        return logits, decoder_target
