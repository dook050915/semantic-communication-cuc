# Transformer 语义通信实验

与 LSTM 共用**同一条信道链**(channel encoder/decoder + 功率归一化 + AWGN/瑞利),仅把语义编解码器从 LSTM 换成 Transformer,做架构对比。

## 配置

- 模型:Transformer 编码器/解码器,d_model = 128 / 256,3 层,8 头,正弦位置编码,因果 + 填充双掩码
- 训练 recipe:warmup(约 10% steps)+ cosine 衰减,峰值 lr 5e-4,grad_clip 1.0,batch 96;50k 用 dropout 0、40 epochs,110k 用 dropout 0.1、30 epochs
- 信道:AWGN / 瑞利,channel_dim = 256(及 16),多 SNR 随机训练 [-10, 20] dB
- 数据:Europarl 50k 与实际 110,844 句的扩展集(目录沿用 `200k` 命名)
- 损失:两组均为 `CrossEntropyLoss(ignore_index=pad_idx)`;当前代码**未启用 label smoothing**

## 结果(AWGN,channel_dim=256)

| SNR (dB) | -10 | -5 | 0 | 10 | 20 |
|---|---|---|---|---|---|
| LSTM (50k) | 0.159 | 0.621 | 0.830 | 0.867 | 0.870 |
| Transformer d256 (50k) | 0.283 | 0.443 | 0.508 | 0.538 | 0.539 |
| Transformer d256 (110k,探索性) | 0.279 | 0.390 | 0.431 | 0.450 | 0.452 |

## 结论

- **严格同口径的 50k 对比**:−10 dB 时 Transformer 更高(0.283 vs 0.159);从 −5 dB 开始 LSTM 明显领先(−5 dB 为 0.621 vs 0.443,20 dB 为 0.870 vs 0.539)。
- **曲线形态**:Transformer 50k 的 BLEU-SNR 曲线明显比 LSTM 平。解码器语言先验较强、信道表征利用不足是候选解释,仍需通过 memory/接收序列遮蔽等消融验证。
- **110k 复跑不能解释数据量效应**:它同时改变了词表(16,471→25,826)、dropout(0→0.1)、训练轮数(40→30)和数据划分;最优 val loss 还出现在最后一轮。因此不能将较低 BLEU 归因于“文本更多”。
- **no-channel 仅是 sanity check**:它说明模型在无信道条件下能够优化但泛化较弱,不能单独证明信道实现无误或根因已经确定。
- 完整分析见根目录 README「核心结果 2」与「踩过的坑」。

## Transformer 特有的坑(详见根 README)

- 两套相反的 mask 约定(信道 True=有效 vs PyTorch True=屏蔽)
- 必须 warmup + cosine + grad_clip,固定低 lr 会欠训练;无 warmup 的高 lr 会被开头大梯度带崩
- d_model 须 ≥ channel_dim,否则 channel encoder 秩瓶颈、浪费一半信道
- Mac MPS 需 `enable_nested_tensor=False`
