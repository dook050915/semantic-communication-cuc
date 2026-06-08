# LSTM AWGN句子重构实验（50k, hidden512，SNR=10dB）

## 实验目的

本实验在 50k 数据量 hidden_dim=512 不变的前提下，加入AWGN信道。

目标是观察：在相同训练数据下，加入噪声后，句子重构性能相比无噪声基线的变化。

当前任务是加入信道噪声的AWGN信道（只在hidden state上加入噪声）：

```text
原始句子 -> LSTM Encoder -> 语义状态 -> AWGN噪声 -> LSTM Decoder -> 重构句子
```

## 数据集

数据来自 Europarl English，从中采样并过滤得到 50,000 条英文句子。

主要过滤规则：

- 保留 token 数量在 4 到 30 之间的句子
- 去除 Europarl 标签行
- 去除空行和括号形式的舞台说明

本次实验使用的 processed 数据文件：

```text
data/processed/europarl_en_50k.txt
```

## 模型

模型使用 LSTM Encoder-Decoder。

Encoder 将输入句子编码为 hidden state 和 cell state；Decoder 使用这些状态重构原句。

当前版本没有 attention。

## 超参数

```text
embedding_dim = 128
hidden_dim = 512
num_layers = 1
batch_size = 96
learning_rate = 1e-3
epochs = 20
snr_db = 10

```

## 实验结果

```text
best_epoch = 10
best_val_loss = 2.4545
test_loss = 2.4398
test_bleu = 0.2856
```

BLEU 分数在 0 到 1 之间，越接近 1 表示模型生成的句子和参考句子越接近。

与不加入噪声 相比：

```text
50k_h512 test_loss = 2.4322, test_BLEU = 0.2867
awgn_50k_h512 test_loss = 2.4398, test_BLEU = 0.2856
```

与不加入噪声版相比，加入噪声后，test_loss略微升高，BLEU略微下降，但下降幅度不大，说明经过训练后，模型有了一定的抗噪声性能，且与无噪声的性能差异较小。

## 结果观察
10 dB 下 AWGN 结果和无噪声结果接近
说明 hidden-level 表征对轻度噪声有一定鲁棒性
## 下一步
下一步要做不同 SNR 扫描，比如 0/5/10/15/20 dB，画 BLEU-SNR 曲线

