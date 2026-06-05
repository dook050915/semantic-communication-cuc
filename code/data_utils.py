import json
import os
import random

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

def tokenize(sentence):
    lst = []
    for word in sentence.lower().split():
        lst.append(word)
    return lst


def is_tag_line(line):
    line = line.strip()
    return line.startswith("<") and line.endswith(">")


def is_stage_direction(line):
    line = line.strip()
    return line.startswith("(") and line.endswith(")")


def load_corpus(path, min_len=4, max_len=30, max_sentences=50000):
    lst = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if is_tag_line(line) or (not line.strip()) or is_stage_direction(line):
                continue

            tokens = tokenize(line)
            if min_len <= len(tokens) <= max_len:
                sentence = " ".join(tokens)
                lst.append(sentence)

            if len(lst) >= max_sentences:
                break

    return lst


def load_corpus_from_dir(dir_path, min_len=4, max_len=30, max_sentences=50000):
    lst = []
    for file in sorted(os.listdir(dir_path)):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(dir_path, file)
        if not os.path.isfile(file_path):
            continue

        remaining = max_sentences - len(lst)
        sentences = load_corpus(
            path=file_path,
            min_len=min_len,
            max_len=max_len,
            max_sentences=remaining,
        )
        lst.extend(sentences)

        if len(lst) >= max_sentences:
            break

    return lst


def save_sentences(sentences, path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    print(f"saved {len(sentences)} sentences to {path}")


def split_sentences(sentences, train_ratio=0.8, val_ratio=0.1, seed=42):
    assert train_ratio + val_ratio < 1
    sentence_copy = sentences.copy()
    random.Random(seed).shuffle(sentence_copy)

    n = len(sentence_copy)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_sentences = sentence_copy[:train_end]
    val_sentences = sentence_copy[train_end:val_end]
    test_sentences = sentence_copy[val_end:]
    return train_sentences, val_sentences, test_sentences


def build_vocab(sentences, min_freq=2):
    word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
    lst_freq = {}

    for sentence in sentences:
        for token in tokenize(sentence):
            if token not in lst_freq:
                lst_freq[token] = 1
            else:
                lst_freq[token] += 1

    lst_freq = {k: v for k, v in lst_freq.items() if v >= min_freq}
    sorted_freq = sorted(lst_freq.items(), key=lambda x: (-x[1], x[0]))

    for token, freq in sorted_freq:
        idx = len(word2idx)
        word2idx[token] = idx
        idx2word[idx] = token

    return word2idx, idx2word


def encode(sentence, word2idx):
    enc = [word2idx["<SOS>"]]
    for token in tokenize(sentence):
        if token in word2idx:
            enc.append(word2idx[token])
        else:
            enc.append(word2idx["<UNK>"])
    enc.append(word2idx["<EOS>"])
    return enc


def decode(ids, idx2word):
    sentence = ""
    for idx in ids:
        idx = int(idx)
        if idx not in (0, 1, 2):
            sentence += idx2word[idx] + " "
    return sentence.strip()


class TextDataset(Dataset):
    def __init__(self, sentences, word2idx):
        self.sentences = sentences
        self.word2idx = word2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        ids = encode(self.sentences[idx], self.word2idx)
        return ids, len(ids)


def collate_fn(batch):
    max_len = 1
    padded_ids, lengths = [], []
    pad_id = 0

    for _, length in batch:
        if length > max_len:
            max_len = length
        lengths.append(length)

    for ids, _ in batch:
        padded_ids.append(ids + [pad_id] * (max_len - len(ids)))

    padded_ids = torch.LongTensor(padded_ids)
    lengths = torch.LongTensor(lengths)
    return padded_ids, lengths


def save_vocab(word2idx, idx2word, path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"word2idx": word2idx, "idx2word": idx2word},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"vocab saved to {path}")


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    word2idx = data["word2idx"]
    idx2word = {int(k): v for k, v in data["idx2word"].items()}
    return word2idx, idx2word
