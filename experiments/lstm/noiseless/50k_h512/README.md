# LSTM 无噪声句子重构实验（50k, hidden512）

## 实验目的

本实验在 50k 数据量不变的前提下，将 LSTM hidden_dim 从 256 增大到 512。

目标是观察：在相同训练数据下，增大模型容量是否能提升句子重构质量。

当前任务仍然是不加入信道噪声的理想信道设置：

```text
原始句子 -> LSTM Encoder -> 语义状态 -> LSTM Decoder -> 重构句子
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

当前版本没有 attention，也没有加入信道噪声。

## 训练设置

```text
embedding_dim = 128
hidden_dim = 512
num_layers = 1
batch_size = 96
learning_rate = 1e-3
epochs = 20
```

## 实验结果

```text
best_epoch = 10
train_loss_best = 1.2393
val_loss_best = 2.4491
test_loss = 2.4322
test_BLEU = 0.2867
```

BLEU 分数在 0 到 1 之间，越接近 1 表示模型生成的句子和参考句子越接近。

与前两组 baseline 相比：

```text
20k_h256 test_loss = 2.7512, test_BLEU = 0.2408
50k_h256 test_loss = 2.6157, test_BLEU = 0.2569
50k_h512 test_loss = 2.4322, test_BLEU = 0.2867
```

在 50k 数据上增大 hidden_dim 后，test loss 和 BLEU 都有进一步改善，说明当前 LSTM baseline 的瓶颈不只是数据量，模型容量也会影响句子重构质量。

不过训练后期出现了更明显的过拟合：best checkpoint 出现在 epoch 10，而训练继续到 epoch 20 时 train loss 持续下降，val loss 反而上升。

## 结论

短句和部分高频模板句子能够完整重构。

```text
reference:
the debate is closed.

prediction:
the debate is closed.
```

```text
reference:
that concludes this item.

prediction:
that concludes this item.
```

部分较长句子的结构和关键词有所改善，但仍然会出现数字、专有名词和具体实体的错误。

```text
reference:
we chose to vote in favour of amendments nos 76 and 19 and against, for example, amendments nos 5 and 7.

prediction:
we chose to vote in favour of amendments nos 1 and 31 amendments nos 31 and 31 amendments to 31 <UNK>
```

```text
reference:
mr president, the situation in mauritania is actually getting worse and worse.

prediction:
mr president, the situation in turkish rail is highly rail and unacceptable.
```

这个结果说明，增大 LSTM hidden_dim 能提升整体重构指标，但当前模型仍难以稳定恢复长句中的具体实体、数字和低频词。

## 下一步

当前无噪声 LSTM baseline 已经完成三组对照：

```text
experiments/lstm/noiseless/20k_h256
experiments/lstm/noiseless/50k_h256
experiments/lstm/noiseless/50k_h512
```

下一步在 Encoder 输出的语义状态上加入 AWGN channel，观察不同 SNR 下的 test loss 和 BLEU 变化。
