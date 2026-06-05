import os
import random

import torch

from data_utils import decode, encode


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_ids, batch_lengths in loader:
        batch_ids = batch_ids.to(device)
        batch_lengths = batch_lengths.cpu()

        optimizer.zero_grad()
        logits, target = model(batch_ids, batch_lengths)

        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)

        loss = criterion(logits_flat, target_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_ids, batch_lengths in loader:
            batch_ids = batch_ids.to(device)
            batch_lengths = batch_lengths.cpu()

            logits, target = model(batch_ids, batch_lengths)

            vocab_size = logits.shape[-1]
            logits_flat = logits.reshape(-1, vocab_size)
            target_flat = target.reshape(-1)

            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item()

    return total_loss / len(loader)


def greedy_decode(model, sentence, word2idx, idx2word, max_len=32):
    model.eval()
    device = next(model.parameters()).device

    ids = encode(sentence, word2idx)
    src_lengths = torch.tensor([len(ids)], dtype=torch.long).cpu()
    src_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_ids, src_lengths)
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
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
