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


def train_one_epoch(model, loader, criterion, optimizer, device, snr_db=None,snr_list=None):
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
            hidden = model.channel(hidden, snr_db=snr_db)
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
