# Transformer 语义通信实验

与 LSTM 共用**同一条信道链**(channel encoder/decoder + 功率归一化 + AWGN/瑞利),仅把语义编解码器从 LSTM 换成 Transformer,做架构对比。

## 配置

- 模型:Transformer 编码器/解码器,d_model = 128 / 256,3 层,8 头,正弦位置编码,因果 + 填充双掩码
- 训练 recipe:warmup(约 10% steps)+ cosine 衰减,峰值 lr 5e-4,grad_clip 1.0,batch 96;50k 用 dropout 0,110k 用 dropout 0.1 + label smoothing 0.1
- 信道:AWGN / 瑞利,channel_dim = 256(及 16),多 SNR 随机训练 [-10, 20] dB
- 数据:Europarl 50k 与 110k

## 结果(AWGN,channel_dim=256)

| SNR (dB) | -10 | -5 | 0 | 10 | 20 |
|---|---|---|---|---|---|
| LSTM (50k) | 0.159 | 0.621 | 0.830 | 0.867 | 0.870 |
| Transformer (50k) | 0.229 | 0.374 | 0.440 | 0.471 | 0.474 |
| Transformer (110k) | 0.279 | 0.390 | 0.431 | 0.450 | 0.452 |

## 结论

- **低 SNR(−10/−5 dB):Transformer 更鲁棒;高 SNR:LSTM 远胜**(0.87 vs 0.45)。两者互补。
- **Transformer 的 BLEU-SNR 曲线偏平**(几乎不随信道质量变),而 LSTM 很陡。指向:**Transformer 解码器作为强语言模型,更多依赖语言先验、而非充分利用过信道的语义信号**,曲线被先验封在 ~0.45;LSTM 解码器更依赖信道,高 SNR 才能冲高。
- **更多数据 + 正则(50k→110k)缓解了过拟合(train/val gap 1.36→~0.6),但天花板未升**(高 SNR 仍 ~0.45)→ 瓶颈是**架构层面的解码行为,不是数据量/过拟合**。
- no-channel 自编码对照(train 0.21 / val 1.26)确认非实现 bug:模型能学但泛化受限。
- 完整分析见根目录 README「核心结果 2」与「踩过的坑」。

## Transformer 特有的坑(详见根 README)

- 两套相反的 mask 约定(信道 True=有效 vs PyTorch True=屏蔽)
- 必须 warmup + cosine + grad_clip,固定低 lr 会欠训练;无 warmup 的高 lr 会被开头大梯度带崩
- d_model 须 ≥ channel_dim,否则 channel encoder 秩瓶颈、浪费一半信道
- Mac MPS 需 `enable_nested_tensor=False`
