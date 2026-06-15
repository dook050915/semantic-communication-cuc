import os
import random
import torch
from data_utils import decode, encode

def set_seed(seed=42):
    """固定随机种子，让数据打乱和模型初始化尽量可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """自动选择训练设备：优先 CUDA，其次 Mac MPS，最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, snr_db=None, snr_list=None):
    """训练一个 epoch：前向、算 loss、反向传播、更新参数。

    batch_ids: [batch_size, seq_len]
    logits: [batch_size, seq_len - 1, vocab_size]
    target: [batch_size, seq_len - 1]
    logits_flat: [batch_size * (seq_len - 1), vocab_size]
    target_flat: [batch_size * (seq_len - 1)]
    """
    model.train()
    total_loss = 0.0

    for batch_ids, batch_lengths in loader:
        batch_ids = batch_ids.to(device)
        batch_lengths = batch_lengths.cpu()

        optimizer.zero_grad()
        if snr_list is None:
            logits, target = model(batch_ids, batch_lengths, snr_db=snr_db)
        else:
            logits, target = model(batch_ids, batch_lengths, snr_db=random.choice(snr_list))

        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        loss = criterion(logits_flat, target_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_loss(model, loader, criterion, device, snr_db=None):
    """在不更新参数的情况下计算平均 loss，用于 train eval 或 val。

    形状变化和 train_one_epoch 相同，只是不做 backward 和 optimizer.step。
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_ids, batch_lengths in loader:
            batch_ids = batch_ids.to(device)
            batch_lengths = batch_lengths.cpu()

            logits, target = model(batch_ids, batch_lengths, snr_db=snr_db)
            vocab_size = logits.shape[-1]
            logits_flat = logits.reshape(-1, vocab_size)
            target_flat = target.reshape(-1)

            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item()

    return total_loss / len(loader)


def greedy_decode(model, sentence, word2idx, idx2word, max_len=32, snr_db=None):
    """用贪心策略推理：每一步选择概率最高的 token 作为下一个词。

    编码后:
        ids: list [seq_len]
        src_ids: LongTensor [1, seq_len]
        src_lengths: LongTensor [1]
    每一步:
        decoder_input: LongTensor [1, 1]
        logits: [1, 1, vocab_size]
        logits[:, -1, :]: [1, vocab_size]
        next_id: Python int
    """
    model.eval()
    device = next(model.parameters()).device

    ids = encode(sentence, word2idx)
    src_lengths = torch.tensor([len(ids)], dtype=torch.long).cpu()
    src_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_ids, src_lengths)
        if model.channel is not None:
            hidden, cell = model.transmit(hidden, cell, snr_db=snr_db)
        decoder_input = torch.tensor(
            [[word2idx["<SOS>"]]], dtype=torch.long, device=device
        )
        generated_ids = []

        for _ in range(max_len):
            logits, (hidden, cell) = model.decoder(decoder_input, hidden, cell)
            next_id = logits[:, -1, :].argmax().item()

            if next_id == word2idx["<EOS>"]:
                break

            generated_ids.append(next_id)
            decoder_input = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return decode(generated_ids, idx2word)

def greedy_decode_batch(model, loader, word2idx, idx2word, max_len=32, snr_db=None):
    """整句版（Seq2Seq）的 batch 贪心解码：整个 test_loader 一次性推理，返回预测句子列表。

    和单句版 greedy_decode 的区别：一次处理一个 batch，用 finished 掩码记录哪些句子已生成 <EOS>，
    全部 finished 时提前停。返回 List[str]。
    """
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    with torch.no_grad():
        for src_ids, lengths in loader:
            src_ids = src_ids.to(device)
            lengths = lengths.cpu()
            hidden, cell = model.encoder(src_ids, lengths)
            if model.channel is not None:
                hidden, cell = model.transmit(hidden, cell, snr_db=snr_db)
            decoder_input = torch.full((src_ids.shape[0], 1), word2idx["<SOS>"], dtype=torch.long, device=device)
            finished = torch.tensor([False]*src_ids.shape[0], dtype=torch.bool, device=device)
            generated_ids = torch.zeros((src_ids.shape[0], max_len), dtype=torch.long, device=device)
            for i in range(max_len):
                logits, (hidden, cell) = model.decoder(decoder_input, hidden, cell)
                next_ids = logits[:, -1, :].argmax(dim=1)
                finished |= (next_ids == word2idx["<EOS>"])
                if finished.all():
                    break
                decoder_input = next_ids.unsqueeze(1)
                generated_ids[:, i] = next_ids
            for gen_id in generated_ids:
                all_preds.append(decode(gen_id, idx2word))

    return all_preds


def greedy_decode_token(model, sentence, word2idx, idx2word, max_len=32, snr_db=None):
    """用贪心策略推理：每一步选择概率最高的 token 作为下一个词。

    编码后:
        ids: list [seq_len]
        src_ids: LongTensor [1, seq_len]
        src_lengths: LongTensor [1]
    每一步:
        decoder_input: LongTensor [1, 1]
        logits: [1, 1, vocab_size]
        logits[:, -1, :]: [1, vocab_size]
        next_id: Python int
    """
    model.eval()
    device = next(model.parameters()).device

    ids = encode(sentence, word2idx)
    src_lengths = torch.tensor([len(ids)], dtype=torch.long).cpu()
    src_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    mask = (src_ids != word2idx["<PAD>"])
    with torch.no_grad():
        outputs = model.encoder(src_ids, src_lengths)
        if model.channel is not None:
            outputs = model.transmit_tokens(outputs, mask, snr_db=snr_db)
        decoder_input = torch.tensor(
            [[word2idx["<SOS>"]]], dtype=torch.long, device=device
        )
        generated_ids = []
        hidden, cell = model.decoder.init_state(1, device)

        for _ in range(max_len):
            logits, (hidden, cell) = model.decoder.step(decoder_input, hidden, cell, outputs, mask)
            next_id = logits[:, -1, :].argmax().item()
            if next_id == word2idx["<EOS>"]:
                break

            generated_ids.append(next_id)
            decoder_input = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return decode(generated_ids, idx2word)

def greedy_decode_batch_token(model, loader, word2idx, idx2word, max_len=32, snr_db=None):
    """token 版（TokenSeq2Seq）的 batch 贪心解码：逐 token 过信道 + attention 解码，整批一次推理。

    和 greedy_decode_batch 的区别：encoder 输出整条序列、走 transmit_tokens（带 mask 的逐 token 信道），
    decoder 用 step + attention 从接收序列取 context。返回 List[str]。
    """
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    with torch.no_grad():
        for src_ids, lengths in loader:
            src_ids = src_ids.to(device)
            lengths = lengths.cpu()
            mask = (src_ids != word2idx["<PAD>"])
            outputs = model.encoder(src_ids, lengths)
            hidden, cell = model.decoder.init_state(src_ids.shape[0], device)
            if model.channel is not None:
                outputs = model.transmit_tokens(outputs, mask, snr_db=snr_db)
            decoder_input = torch.full((src_ids.shape[0], 1), word2idx["<SOS>"], dtype=torch.long, device=device)
            finished = torch.tensor([False]*src_ids.shape[0], dtype=torch.bool, device=device)
            generated_ids = torch.full((src_ids.shape[0], max_len), word2idx["<PAD>"], dtype=torch.long, device=device)

            for i in range(max_len):
                logits, (hidden, cell) = model.decoder.step(decoder_input, hidden, cell, outputs, mask)
                next_ids = logits[:, -1, :].argmax(dim=1)
                finished |= (next_ids == word2idx["<EOS>"])
                if finished.all():
                    break
                decoder_input = next_ids.unsqueeze(1)
                generated_ids[:, i] = next_ids
            for gen_id in generated_ids:
                all_preds.append(decode(gen_id, idx2word))

    return all_preds


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, vocab_path, path):
    """保存模型、优化器、epoch、loss、config 和词表路径。"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
        "vocab_path": vocab_path,
    }
    torch.save(checkpoint, path)
    print(f"checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, map_location="cpu"):
    """读取 checkpoint，并把模型参数和可选优化器参数加载回来。"""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

Predict = {"predict_token":greedy_decode_token,
          "predict_token_batch":greedy_decode_batch_token,
          "predict_seq":greedy_decode,
          "predict_seq_batch":greedy_decode_batch,
          }
