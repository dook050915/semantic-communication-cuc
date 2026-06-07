# LSTM 无噪声句子重构实验（50k, hidden256）

## 实验目的

本实验在 20k baseline 的基础上，将训练语料扩大到 50,000 条句子，其他主要超参数保持不变。

目标是观察：在相同 LSTM 模型容量下，增加训练数据是否能提升句子重构质量。
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

## 超参数

```text
embedding_dim = 128
hidden_dim = 256
num_layers = 1
batch_size = 96
learning_rate = 1e-3
epochs = 20
```

## 实验结果

```text
best_epoch = 14
train_loss_best = 1.6084
val_loss_best = 2.6322
test_loss = 2.6157
test_BLEU = 0.2569
```

BLEU 分数在 0 到 1 之间，越接近 1 表示模型生成的句子和参考句子越接近。

与 20k baseline 相比：

```text
20k_h256 test_loss = 2.7512, test_BLEU = 0.2408
50k_h256 test_loss = 2.6157, test_BLEU = 0.2569
```

增加训练数据后，val/test loss 和 BLEU 均有改善，但提升幅度有限。这说明更多数据对句子重构有帮助，但当前 LSTM baseline 的重构质量仍然受到模型结构和 greedy decoding 的限制。

## 结果观察

短句和高频模板句子仍然能较好重构。

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

较长句子仍然容易出现语义漂移，低频词和专有名词仍然容易被映射成 `<UNK>` 或被替换成其他常见表达。

```text
reference:
fire-fighters who indulge in arson do not deserve our trust.

prediction:
<UNK> who do not <UNK> in our <UNK> friends on.
```

```text
reference:
mr president, the situation in mauritania is actually getting worse and worse.

prediction:
mr president, the situation in zimbabwe is already in brussels and practical.
```

这个结果说明，单纯增加数据量可以改善 loss 和部分重构效果，但对长句语义保持、低频词处理和专有名词恢复的提升有限。

## 下一步

在相同 50k 数据上增大模型容量，继续测试：

```text
lstm_noiseless_50k_h512
```

如果 hidden_dim=512 能带来更明显的 BLEU 提升，说明当前瓶颈部分来自模型容量；如果提升仍然有限，则后续应重点考虑 attention、Transformer 或更好的词表/解码策略。
