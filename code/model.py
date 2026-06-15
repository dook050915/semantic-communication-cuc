import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from channel import power_normalize, sequence_power_normalize

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
    """完整 Seq2Seq 模型：Encoder 读原句，可选信道扰动 hidden，Decoder 重构原句。"""

    def __init__(self, encoder, decoder, channel=None, channel_dim=128):
        """把已经定义好的 encoder 和 decoder 组合起来，并建立信道编解码线性层。"""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.channel = channel
        L = encoder.lstm.num_layers
        H = encoder.lstm.hidden_size
        state_dim = 2*L*H
        self.channel_encoder = nn.Linear(state_dim, channel_dim)
        self.channel_decoder = nn.Linear(channel_dim, state_dim)

    def transmit(self, hidden, cell, snr_db=None):
        """发送链:hidden,cell [L,B,H] → 编码→归一化→加噪→解码 → 还原 [L,B,H]"""
        L, B, H = hidden.shape
        hidden = hidden.permute(1,0,2).reshape(B,-1)
        cell = cell.permute(1,0,2).reshape(B,-1)
        state = torch.cat((hidden,cell),dim=1)
        state = self.channel_encoder(state)
        state = power_normalize(state)
        state_channel = self.channel(state, snr_db) if snr_db is not None else state
        state_hat = self.channel_decoder(state_channel)
        hidden_hat = state_hat[:,:L*H].reshape(B,L,H).permute(1,0,2).contiguous()
        cell_hat = state_hat[:,L*H:].reshape(B,L,H).permute(1,0,2).contiguous()
        return hidden_hat, cell_hat

    def forward(self, src_ids, src_lengths, snr_db=None):
        """训练时使用 teacher forcing，返回 logits 和右移后的 target。
            snr_db: AWGN 信道的信噪比；channel=None 时不使用
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
        if self.channel is not None:
            hidden, cell = self.transmit(hidden, cell, snr_db)
        decoder_input = src_ids[:, :-1]
        decoder_target = src_ids[:, 1:]
        logits, _ = self.decoder(decoder_input, hidden, cell)
        return logits, decoder_target

class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) Attention 层：用 tanh(W_k·k + W_q·q) 打分，再加权求 context。"""

    def __init__(self, key_dim, query_dim, hidden_dim):
        """定义把 key/query 投到打分空间的三个线性层（均无 bias）。"""
        super().__init__()
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self,keys,values,queries,mask=None):
        '''
        输入:
            keys: [B, T, H]
            values: [B, T, H]
            queries: [B, H]
        
        输出:
            scores: [B, T]
            context
        '''
        keys = self.W_k(keys)
        queries = self.W_q(queries).unsqueeze(1)
        scores = self.W_v(torch.tanh(keys + queries)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        self.attention_weights = torch.softmax(scores, dim=-1).unsqueeze(1)
        context = torch.bmm(self.attention_weights, values).squeeze(1)
        return context

class TokenEncoder(nn.Module):
    """Token-level Encoder。

    目标:
        输入 src_ids [B, T]
        输出 encoder_outputs [B, T, H]

    和旧 Encoder 的区别:
        旧 Encoder 返回最后的 hidden/cell。
        这里返回每个 token 位置的 LSTM output 序列。
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, pad_idx=0):
        """定义 embedding 层和 LSTM 层（与旧 Encoder 相同，只是 forward 返回整条序列）。"""
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
        """输入源句子 id 和真实长度，输出每个 token 位置的 LSTM output 序列 [B, T, H]。"""
        embedded = self.embedding(src_ids)
        packed_embedded = pack_padded_sequence(
            input=embedded,
            lengths=src_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        outputs, _ = self.lstm(packed_embedded)
        outputs,_ = pad_packed_sequence(
            outputs, 
            batch_first=True,
            total_length=src_ids.size(1),
        )
        return outputs   


class TokenAttentionDecoder(nn.Module):
    """Token-level Attention Decoder。

    目标:
        输入 decoder token 和 received_outputs。
        每一步用 attention 从 received_outputs 里取 context。
        输出 logits [B, T_tgt, vocab_size]。
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, pad_idx=0, attention_hdim=128):
        """定义 embedding、attention、LSTM（输入拼了 context 所以是 embed_dim+hidden_dim）和输出层。"""
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = AdditiveAttention(hidden_dim, hidden_dim, attention_hdim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, batch_size, device):
        """初始化 decoder LSTM 的 hidden/cell 为全零（attention 取信息不靠初始状态）。"""
        L = self.lstm.num_layers
        H = self.lstm.hidden_size
        hidden = torch.zeros((L, batch_size, H), device=device)
        cell = torch.zeros((L, batch_size, H), device=device)
        return (hidden, cell)

    def step(self, input_ids, hidden, cell, received_outputs, src_mask):
        """单步解码：用上一层 hidden 作 query 从接收序列取 context，拼 embedding 喂 LSTM，输出当前 logits。"""
        queries = hidden[-1]
        embedded = self.embedding(input_ids)
        context = self.attention(received_outputs, received_outputs,queries,src_mask)
        decoder_inputs = torch.cat((embedded, context.unsqueeze(1)), dim=-1)
        logits_t, (hidden, cell) = self.lstm(decoder_inputs, (hidden, cell))
        logits_t = self.fc(logits_t)
        return logits_t, (hidden, cell)


    def forward(self, tgt_input_ids, received_outputs, src_mask):
        """训练时使用 teacher forcing 解码。

        输入:
            tgt_input_ids: [B, T_tgt]
            received_outputs: [B, T_src, H]
            src_mask: [B, T_src],真实 token 为 True,PAD 为 False

        输出:
            logits: [B, T_tgt, vocab_size]
        """
        T_tgt = tgt_input_ids.size(1) 
        L = self.lstm.num_layers
        H = self.lstm.hidden_size
        B = tgt_input_ids.size(0)
        device = tgt_input_ids.device
        hidden, cell = self.init_state(B, device)
        logits = torch.zeros((B, T_tgt, self.fc.out_features), device=device)
        for t in range(T_tgt):
            logits_t, (hidden, cell) = self.step(tgt_input_ids[:, t:t+1], hidden, cell, received_outputs, src_mask)
            logits[:, t, :] = logits_t.squeeze(1)
        return logits


class TokenSeq2Seq(nn.Module):
    """Token-level Seq2Seq 总模型。

    总链路:
        src_ids [B,T]
        -> TokenEncoder
        -> encoder_outputs [B,T,H]
        -> token-level channel
        -> received_outputs [B,T,H]
        -> TokenAttentionDecoder
        -> logits [B,T-1,V]
    """

    def __init__(self, encoder, decoder, channel=None, channel_dim=128):
        """组合 token 级 encoder/decoder，并建立逐 token 的信道编解码线性层。"""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.channel = channel
        state_dim = encoder.lstm.hidden_size
        self.channel_encoder = nn.Linear(state_dim, channel_dim)
        self.channel_decoder = nn.Linear(channel_dim, state_dim)

    def transmit_tokens(self, encoder_outputs, mask, snr_db=None):
        """把 encoder_outputs [B,T,H] 逐 token 送过信道。"""
        channel_symbols = self.channel_encoder(encoder_outputs)
        channel_norm = sequence_power_normalize(channel_symbols, mask)
        if snr_db is not None:
            x = self.channel(channel_norm, snr_db)
        else:
            x = channel_norm
        received_outputs = self.channel_decoder(x)
        return received_outputs

    def forward(self, src_ids, src_lengths, snr_db=None):
        """使用 teacher forcing 的训练前向传播。"""
        mask = (src_ids != self.encoder.embedding.padding_idx)
        encoder_outputs = self.encoder(src_ids, src_lengths)
        if self.channel is not None:
            received_outputs = self.transmit_tokens(encoder_outputs, mask, snr_db)
        else:
            received_outputs = encoder_outputs
        decoder_input = src_ids[:, :-1]
        decoder_target = src_ids[:, 1:]
        logits = self.decoder(decoder_input, received_outputs, mask)

        return logits, decoder_target

Encoders = {"TokenEncoder":TokenEncoder,
            "Encoder":Encoder}

Decoders = {"TokenAttentionDecoder":TokenAttentionDecoder,
            "Decoder":Decoder}

Models = {"TokenSeq2Seq":TokenSeq2Seq,
            "Seq2Seq":Seq2Seq}