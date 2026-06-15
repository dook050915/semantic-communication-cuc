import json
import os
import sacrebleu
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import (
    TextDataset,
    collate_fn,
    load_vocab,
    split_sentences,
)
from model import Encoder, Decoder, Seq2Seq
from train_lstm import config, load_training_sentences, resolve_path
from train_utils import evaluate_loss, greedy_decode, load_checkpoint, get_device, set_seed
from channel import channel_types


def compute_bleu(references, predictions):
    """语料级 BLEU,改用 sacrebleu 替换原手写实现。

    入参不变:references / predictions 都是 List[str](已小写、空格分词的句子)。
    返回:0-1 区间 float(sacrebleu 原生是 0-100,这里 /100,对齐旧数值与 DeepSC 论文刻度)。
    """
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="none")
    return bleu.score / 100.0

def save_results(results, path):
    """把评测结果字典存成 json 文件（自动建目录）。"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"results saved to {path}")


def save_predictions(references, predictions, path, max_samples=1000000):
    """把 reference/prediction 成对写入文本文件，最多写 max_samples 条，便于人工抽查译文。"""
    assert len(references) == len(predictions)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for ref, pred in zip(references, predictions):
            count += 1
            f.write(f"sample {count}\nreference: {ref}\nprediction: {pred}\n{'='*50}\n")
            if count >= max_samples:
                break
    print(f"predictions saved to {path} with {count} samples")

def main():
    """评测入口（整句 Seq2Seq）：加载 best checkpoint，在 test 集上算 loss + BLEU 并落盘。"""
    device = get_device()
    eval_config = config.copy()
    set_seed(eval_config["seed"])
    eval_config['results_path'] = "experiments/lstm/awgn/hidden_cell/multi_snr_50k_h512/results.json"
    eval_config['prediction_samples_path'] = "experiments/lstm/awgn/hidden_cell/multi_snr_50k_h512/prediction_samples.txt"
    eval_config['max_samples'] = 100

    if eval_config["use_channel"]:
        channel = channel_types[eval_config["channel_type"]]()
    else:
        channel = None

    sentences = load_training_sentences(eval_config)
    train_sentences, val_sentences, test_sentences = split_sentences(sentences, train_ratio=eval_config["train_ratio"], val_ratio=eval_config["val_ratio"], seed=eval_config["seed"])

    vocab_path = resolve_path(eval_config["vocab_path"])
    word2idx, idx2word = load_vocab(str(vocab_path))
    vocab_size = len(word2idx)
    pad_idx = word2idx["<PAD>"]

    test_dataset = TextDataset(test_sentences, word2idx)
    test_loader = DataLoader(test_dataset, batch_size=eval_config["batch_size"], shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(vocab_size, eval_config["embed_dim"], eval_config["hidden_dim"], eval_config["num_layers"], pad_idx)
    decoder = Decoder(vocab_size, eval_config["embed_dim"], eval_config["hidden_dim"], eval_config["num_layers"], pad_idx)
    model = Seq2Seq(encoder, decoder, channel)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_ckpt_path = resolve_path(eval_config["best_checkpoint_path"])
    checkpoint = load_checkpoint(model, None, str(best_ckpt_path), map_location=str(device))
    test_loss = evaluate_loss(model, test_loader, criterion, device, snr_db=eval_config["snr_db"])

    references = []
    predictions = []
    for sentence in test_sentences:
        references.append(sentence)
        pred = greedy_decode(model, sentence, word2idx, idx2word, max_len=eval_config["max_len"]+2, snr_db=eval_config["snr_db"])
        predictions.append(pred)
    test_bleu = compute_bleu(references, predictions)
    print(f"test_loss: {test_loss:.4f}")
    print(f"test_bleu: {test_bleu:.4f}")

    results = {
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_val_loss': checkpoint['val_loss'],
        'train_loss_best': checkpoint['train_loss'],
        'test_loss': test_loss,
        'test_bleu': test_bleu,
        'num_test_sentences': len(test_sentences),
        "eval_config": eval_config,
    }
    save_results(results, resolve_path(eval_config['results_path']))
    save_predictions(references, predictions, resolve_path(eval_config['prediction_samples_path']), max_samples=eval_config["max_samples"])


if __name__ == "__main__":
    main()
