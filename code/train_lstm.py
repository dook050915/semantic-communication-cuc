from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from channel import channel_types

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
    "data_path": "data/processed/europarl_en_50k.txt",
    "raw_data_dir": "data/raw/europarl-v7/txt/en",
    "max_sentences": 50000,
    "min_len": 4,
    "max_len": 30,
    "min_freq": 2,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "embed_dim": 128,
    "hidden_dim": 512,
    "channel_dim": 128,
    "attention_hdim": 128,
    "num_layers": 1,
    "batch_size": 96,
    "lr": 1e-3,
    "epochs": 20,
    "use_channel":True,
    "channel_type":"Rayleigh",
    "encoder":"TokenEncoder",
    "decoder":"TokenAttentionDecoder",
    "Predict":"predict_token",
    "model":"TokenSeq2Seq",
    "snr_db":10,
    "snr_list":[-10, -5, 0, 5, 10, 15, 20],
    "vocab_path": "experiments/lstm/Rayleigh/tokenSeq2Seq/multi_snr_50k_h512_c16_a128/vocab.json",
    "checkpoint_path": "experiments/lstm/Rayleigh/tokenSeq2Seq/multi_snr_50k_h512_c16_a128/checkpoint_epoch20.pt",
    "best_checkpoint_path": "experiments/lstm/Rayleigh/tokenSeq2Seq/multi_snr_50k_h512_c16_a128/checkpoint_best.pt",
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
        pad_idx=pad_idx,
    )
    decoder = Decoders[config["decoder"]](
        vocab_size=vocab_size,
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        pad_idx=pad_idx,
        attention_hdim=config["attention_hdim"]
    )
    model = Models[config["model"]](encoder, decoder,channel=channel,channel_dim=config["channel_dim"]).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_loss = float("inf")
    best_checkpoint_path = resolve_path(config["best_checkpoint_path"])

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,config["snr_db"],config["snr_list"])
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

    print("\nGreedy decode samples:")
    for sentence in train_sentences[:3]:
        pred = Predict[config["Predict"]](model, sentence, word2idx, idx2word, max_len=32,snr_db=config["snr_db"])
        print("src :", sentence)
        print("pred:", pred)
        print()


if __name__ == "__main__":
    main()
