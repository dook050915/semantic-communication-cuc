# LSTM 无噪声句子重构实验（20k）

## 实验目的

本实验先不加入信道噪声，只验证最基础的句子重构链路是否能跑通。
当前任务可以理解为理想信道条件下的语义传递：

```text
原始句子 -> LSTM Encoder -> 语义状态 -> LSTM Decoder -> 重构句子
```

## 数据集

数据来自 Europarl English，从中采样并过滤得到 20,000 条英文句子。
主要过滤规则：

- 保留 token 数量在 4 到 30 之间的句子
- 去除 Europarl 标签行
- 去除空行和括号形式的舞台说明

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
best_epoch = 17
train_loss_best = 1.7667
val_loss_best = 2.8219
test_loss = 2.7512
test_BLEU = 0.2408
```

BLEU 分数在 0 到 1 之间，越接近 1 表示模型生成的句子和参考句子越接近。

train_loss 明显低于 val_loss，说明模型已经出现一定过拟合。

## 结果观察

短句和高频模板句子的重构效果较好。

```text
reference:
i declare adjourned the session of the european parliament.

prediction:
i declare adjourned the session of the european parliament.
```

长句更容易出现语义漂移，低频词和专有名词容易被映射成 `<UNK>`。

```text
reference:
b5­0063/2001 by mrs hautala, mrs maes, mr gahrton and mrs mckenna, on behalf of the group of the greens/european free alliance, on the law on the khmer rouge trial;

prediction:
<UNK> by mrs <UNK> mrs sanders-ten holte, and mrs <UNK> on behalf of the group of the european liberal, democrat and reform party, on the <UNK> on <UNK> .
```

这个结果说明当前 LSTM baseline 已经具备一定的句子重构能力，但对长句、低频词和专有名词的处理还比较弱。

## 下一步

后续在无噪声设置下继续扩展了两组对照实验：

```text
experiments/lstm/noiseless/50k_h256
experiments/lstm/noiseless/50k_h512
```

这两组实验分别用于观察数据量和模型容量对句子重构质量的影响。

完成无噪声 baseline 对照后，再在 Encoder 输出的语义状态上加入 AWGN channel，观察不同 SNR 下的 test loss 和 BLEU 变化。
