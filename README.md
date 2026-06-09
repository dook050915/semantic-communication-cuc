# Semantic Communication — CUC

文本语义通信项目:复现 DeepSC(Xie et al., IEEE TSP 2021)的核心思路。先用 LSTM 搭出能跑通的端到端 baseline 并加入信道,再逐步升级到 Transformer 与完整物理信道。

状态:LSTM baseline 已跑通(无信道 + AWGN 信道,已产出 BLEU-SNR 曲线),正在准备 Transformer(DeepSC)版本。

---

## 项目目标

把传统通信范式(Shannon Level A,精确传输 bit)换成语义层(Level B,传输含义),研究神经网络能否在带噪信道下学到比传统「信源编码 + 信道编码 + 调制」更鲁棒的端到端文本传输策略。第一版聚焦把整条链路跑通,并产出 BLEU 随 SNR 变化的证据曲线。

## 路线与进度

方法路线(详见 [roadmap.md](./roadmap.md)):LSTM baseline → 加信道 → 升级 Transformer。

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1. LSTM baseline(无信道) | Seq2Seq 自编码,端到端重建句子,BLEU 评估 | 完成 |
| 2. LSTM + AWGN 信道 | 对语义状态加噪,SNR 扫描,BLEU-SNR 曲线,鲁棒性消融 | 完成 |
| 3. Transformer(DeepSC) | 编解码升级为 Transformer,补全 channel encoder/decoder + 功率归一化,对比两版;后续加 Rayleigh | 进行中 |

简化策略:第一版不引入 MI 正则项、BERT 预训练 embedding、Transformer,先用 LSTM + 纯交叉熵 loss + 从零训练的词向量把链路跑通,这些增强项留到第二版。

## 系统模型

第一版(LSTM,已实现):

```
源文本 s
  → Embedding + LSTM Encoder      → 语义状态 (hidden, cell)
  → AWGN 信道(对语义状态加噪)     → 带噪状态
  → LSTM Decoder + Linear         → 重建文本 ŝ

评估:BLEU(s, ŝ) 随 SNR ∈ [-10, 20] dB 变化
```

目标架构(DeepSC,第二版):

```
源文本 s
  → Semantic Encoder   ← Transformer Encoder
  → Channel Encoder    ← FC,投影到受控维度的发送符号 + 功率归一化
  → Physical Channel   ← AWGN / Rayleigh:  y = h·x + n
  → Channel Decoder    ← FC
  → Semantic Decoder   ← Transformer Decoder
  → 重建文本 ŝ
```

## 实验

数据:Europarl 英文语料,过滤保留 4–30 词的句子,采样 50k(及 20k 对照)。train / val / test = 8 / 1 / 1,固定随机种子。

已完成的实验(每组结果见对应目录下的 README、results.json 与曲线):

- `experiments/lstm/noiseless/` — 无信道 baseline,对照数据量(20k / 50k)与模型容量(h256 / h512)
- `experiments/lstm/awgn/hidden_only/` — AWGN 加在 hidden state,固定 10 dB 训练 vs 多 SNR 训练
- `experiments/lstm/awgn/hidden_cell/` — AWGN 同时加在 hidden 与 cell state

主要观察:

- BLEU 随 SNR 升高而上升,符合「信道质量↑ → 语义恢复↑」的预期(固定 SNR 训练的模型最明显)
- 多 SNR 训练显著提升低 SNR 条件下的鲁棒性
- cell state 对 LSTM 重构至关重要:同时扰动 hidden 与 cell 时,低 SNR 下退化明显加重

BLEU 用 sacrebleu(corpus 级)计算,与 DeepSC 等文献可比。BLEU-SNR 对比曲线:`experiments/lstm/awgn/bleu_snr_sacrebleu.png`

## 关键文献

1. Xie, Qin, Li, Juang. *Deep Learning Enabled Semantic Communication Systems*. IEEE TSP 2021.(DeepSC)
2. O'Shea & Hoydis. *An Introduction to Deep Learning for the Physical Layer*. IEEE TCCN 2017.

精读笔记见 [notes/](./notes/)。

## 代码结构

```
.
├── README.md
├── roadmap.md       阶段规划(项目执行地图)
├── code/            数据、模型、训练、评估、信道
├── notes/           文献精读笔记
├── experiments/     实验结果,按「模型 / 信道 / 状态扰动方式 / 配置」组织
└── specs/           各阶段任务说明
```

技术栈:Python 3.10 · PyTorch · matplotlib。纯本地、免费、CPU 可跑通。

## Author

杜可正 — 中国传媒大学 信息与通信工程学院
202311103060@mails.cuc.edu.cn
</content>
</invoke>
