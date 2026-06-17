from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from channel import channel_types
import math
from data_utils import (
    TextDataset,
    build_vocab,
    collate_fn,
    load_corpus,
    load_corpus_from_dir,
    save_vocab,
    split_sentences,
)
from model import Encoders, Decoders, Models
from train_utils import (
    evaluate_loss,
    get_device,
    Predict,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


config = {
    "seed": 42,
    "data_path": "data/processed/europarl_en_200k.txt",
    "raw_data_dir": "data/raw/europarl-v7/txt/en",
    "max_sentences": 200000,
    "min_len": 4,
    "max_len": 30,
    "min_freq": 2,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "embed_dim": 256,
    "hidden_dim": 512,
    "channel_dim": 256,
    "num_layers": 3,
    "nhead": 8,
    "batch_size": 96,
    "lr": 5e-4,
    "warmup_ratio": 0.1,
    "grad_clip": 1.0,
    "epochs": 30,
    "dropout": 0.1,
    "use_channel":True,
    "channel_type":"AWGN",
    "encoder":"TransformerEncoder",
    "decoder":"TransformerDecoder",
    "Predict":"predict_transformer_batch",
    "model":"TransformerSeq2Seq",
    "snr_db":10,
    "snr_list":[-10, -5, 0, 5, 10, 15, 20],
    "vocab_path": "experiments/transformer/AWGN/multi_snr_200k_d256_h8_L3_drop1_lr5e4_warmup10/vocab.json",
    "checkpoint_path": "experiments/transformer/AWGN/multi_snr_200k_d256_h8_L3_drop1_lr5e4_warmup10/checkpoint_epoch30.pt",
    "best_checkpoint_path": "experiments/transformer/AWGN/multi_snr_200k_d256_h8_L3_drop1_lr5e4_warmup10/checkpoint_best.pt",
}


def resolve_path(path):
    """把配置里的相对路径转成基于项目根目录的绝对路径。"""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_training_sentences(config):
    """优先读取 processed 数据；如果没有，就从 raw Europarl 目录现场构建。"""
    data_path = resolve_path(config["data_path"])
    if data_path.exists():
        return load_corpus(
            str(data_path),
            min_len=config["min_len"],
            max_len=config["max_len"],
            max_sentences=config["max_sentences"],
        )

    raw_data_dir = resolve_path(config["raw_data_dir"])
    return load_corpus_from_dir(
        str(raw_data_dir),
        min_len=config["min_len"],
        max_len=config["max_len"],
        max_sentences=config["max_sentences"],
    )


def main():
    """完整训练入口：加载数据、建词表、建模型、训练、保存、预测样例。"""
    device = get_device()
    config["device"] = str(device)
    set_seed(config["seed"])
    print("device:", device)

    if config["use_channel"]:
        channel = channel_types[config["channel_type"]]()
    else:
        channel = None
    
    sentences = load_training_sentences(config)
    print("num sentences:", len(sentences))

    train_sentences, val_sentences, test_sentences = split_sentences(
        sentences,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )
    
    word2idx, idx2word = build_vocab(train_sentences, min_freq=config["min_freq"])
    vocab_path = resolve_path(config["vocab_path"])
    save_vocab(word2idx, idx2word, str(vocab_path))

    train_dataset = TextDataset(train_sentences, word2idx)
    val_dataset = TextDataset(val_sentences, word2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    vocab_size = len(word2idx)
    pad_idx = word2idx["<PAD>"]
    config["vocab_size"] = vocab_size
    config["pad_idx"] = pad_idx

    encoder = Encoders[config["encoder"]](
        vocab_size=vocab_size,
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        nhead=config["nhead"],
        pad_idx=pad_idx,
        dropout=config["dropout"],
    )
    decoder = Decoders[config["decoder"]](
        vocab_size=vocab_size,
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        pad_idx=pad_idx,
        nhead=config["nhead"],
        dropout=config["dropout"],
    )
    model = Models[config["model"]](encoder, decoder, channel=channel, channel_dim=config["channel_dim"]).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = int(config["warmup_ratio"] * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_checkpoint_path = resolve_path(config["best_checkpoint_path"])

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,config["snr_db"],config["snr_list"], scheduler, config["grad_clip"])
        val_loss = evaluate_loss(model, val_loader, criterion, device,config["snr_db"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
                vocab_path=str(vocab_path),
                path=str(best_checkpoint_path),
            )

        print(
            f"Epoch {epoch + 1}/{config['epochs']}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

    checkpoint_path = resolve_path(config["checkpoint_path"])
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config["epochs"],
        train_loss=train_loss,
        val_loss=val_loss,
        config=config,
        vocab_path=str(vocab_path),
        path=str(checkpoint_path),
    )



if __name__ == "__main__":
    main()
