import math
from collections import Counter
import json
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import (
    TextDataset,
    collate_fn,
    load_vocab,
    split_sentences,
    tokenize,
)
from model import Encoder, Decoder, Seq2Seq
from train_lstm import config, load_training_sentences, resolve_path
from train_utils import evaluate_loss, greedy_decode, load_checkpoint, get_device, set_seed

def get_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        n_gram = tuple(tokens[i:i+n])
        ngrams.append(n_gram)
    return Counter(ngrams)


def compute_bleu(references, predictions, max_n=4, smooth=1e-9):
    assert len(references) == len(predictions)
    matched_counts = [0] * max_n
    total_counts = [0] * max_n

    pred_len_total = 0
    ref_len_total = 0

    for ref, pred in zip(references, predictions):
        ref_tokens = tokenize(ref)
        pred_tokens = tokenize(pred)
        pred_len_total += len(pred_tokens)
        ref_len_total += len(ref_tokens)
        for n in range(1, max_n + 1):  
            ref_ngrams = get_ngrams(ref_tokens, n)
            pred_ngrams = get_ngrams(pred_tokens, n) 
            total_counts[n-1] += sum(pred_ngrams.values())
            for pred_gram, pred_count in pred_ngrams.items():
                if pred_gram not in ref_ngrams:
                    continue
                ref_count = ref_ngrams[pred_gram]
                matched_counts[n-1] += min(pred_count, ref_count)
    precisions = []
    for i in range(max_n):
        
        if total_counts[i] == 0 or matched_counts[i] == 0:
            precision = smooth
        else:
            precision = matched_counts[i] / total_counts[i]       
        precisions.append(precision)

    log_precision_sum = sum(math.log(p) for p in precisions)
#brevity penalty
    if pred_len_total == 0:
        return 0.0
    elif pred_len_total > ref_len_total:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len_total / pred_len_total)

    bleu = bp * math.exp(log_precision_sum / max_n)
    return bleu

def save_results(results, path):
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
    device = get_device()
    eval_config = config.copy()
    set_seed(eval_config["seed"])
    eval_config['results_path'] = "experiments/lstm_noiseless_20k/results.json"
    eval_config['prediction_samples_path'] = "experiments/lstm_noiseless_20k/prediction_samples.txt"
    eval_config['max_samples'] = 100

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
    model = Seq2Seq(encoder, decoder)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_ckpt_path = resolve_path(eval_config["best_checkpoint_path"])
    checkpoint = load_checkpoint(model, None, str(best_ckpt_path), map_location=str(device))
    print(f"epoch: {checkpoint['epoch']}")
    print(f"val_loss: {checkpoint['val_loss']:.4f}")

    test_loss = evaluate_loss(model, test_loader, criterion, device)

    references = []
    predictions = []
    for sentence in test_sentences:
        references.append(sentence)
        pred = greedy_decode(model, sentence, word2idx, idx2word, max_len=eval_config["max_len"]+2)
        predictions.append(pred)
    test_bleu = compute_bleu(references, predictions)
    print(f"test_loss: {test_loss:.4f}")
    print(f"test_bleu: {test_bleu:.4f}")

    results = {
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_val_loss': checkpoint['val_loss'],
        'test_loss': test_loss,
        'test_bleu': test_bleu,
        'num_test_sentences': len(test_sentences),
        "eval_config": eval_config,
    }
    save_results(results, resolve_path(eval_config['results_path']))
    save_predictions(references, predictions, resolve_path(eval_config['prediction_samples_path']),max_samples=eval_config["max_samples"])


if __name__ == "__main__":
    main()