# Semantic Communication — CUC

> Text-domain semantic communication for 6G, replicating **DeepSC** (Xie et al., IEEE TSP 2021) and extending it to physical-layer fading channels.

![Status](https://img.shields.io/badge/status-WIP-orange) ![Phase](https://img.shields.io/badge/phase-1%2F3%20%28LSTM%20baseline%29-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 项目目标

把传统通信范式（Shannon **Level A** — 精确传输 bit）替换为语义层（**Level B** — 传输含义），研究在 **AWGN / Rayleigh** 衰落信道下，神经网络是否能学到比 `Source Coding + Channel Coding + QAM` 更鲁棒的端到端传输策略。

## 🛣️ 三阶段路线

| 阶段 | 内容 | 目标 | 当前状态 |
|------|------|------|---------|
| **① LSTM Baseline** | Seq2Seq + 手动 FC 层模拟基带 | 跑通端到端流程，验证 fra-eng 数据可解码 | 🔧 进行中 |
| **② Transformer (DeepSC)** | 编码器/解码器换 Transformer + CE Loss | BLEU 对比 LSTM，复现论文核心结论 | ⏳ 待启动 |
| **③ Physical Channel** | 嵌入 AWGN + Rayleigh 衰落 `y = hx + n` | 产出 **SNR vs. BLEU** 曲线，对比传统方案 | ⏳ 待启动 |

> **简化策略**：考虑到时间窗口（6 月底前需出第一版结果），暂不引入 BERT 预训练 embedding 和 MI（互信息）正则项，先用纯 CE Loss + 词向量 from scratch 跑通。

## 📐 系统模型示意

```
源文本 s ──┐
           ↓
        [Semantic Encoder]    ←  Transformer Encoder
           ↓ z (语义向量)
        [Channel Encoder]      ←  FC layers
           ↓ x (基带符号)
        [Physical Channel]     ←  AWGN / Rayleigh:  y = h·x + n
           ↓ y
        [Channel Decoder]      ←  FC layers
           ↓ ẑ
        [Semantic Decoder]     ←  Transformer Decoder
           ↓
        重建文本 ŝ
        
评估指标：BLEU(s, ŝ)  vs.  SNR ∈ [-6, 18] dB
```

## 📚 关键文献

1. **[DeepSC]** Xie, Qin, Li, Juang. *Deep Learning Enabled Semantic Communication Systems*. IEEE TSP 2021.
2. **[E2E PHY]** O'Shea & Hoydis. *An Introduction to Deep Learning for the Physical Layer*. IEEE TCCN 2017.
3. **[Theory]** Bao, Basu, et al. *Towards a Theory of Semantic Communication*. IEEE NSW 2011.
4. **[Survey]** Shi et al. *Semantic Communications: Principles and Challenges*. 2021.

📖 精读笔记：见 [`notes/`](./notes/)

## 🗓️ Roadmap

- [x] 仓库 scaffold（2026-05-24）
- [ ] 精读 DeepSC 论文 + 整理笔记（5/27-5/28）
- [ ] fra-eng 数据集预处理 + Vocab 构建（5/30）
- [ ] LSTM Baseline 训练 + BLEU 评估（6/3-6/8）
- [ ] Transformer 实现 + 论文对比实验（6/10-6/20）
- [ ] AWGN + Rayleigh 信道嵌入 + SNR 扫描（6/22-6/30）

## 🛠️ Tech Stack

`Python 3.10` · `PyTorch 2.x` · `NumPy` · `nltk` (BLEU) · `matplotlib`

## 📂 仓库结构

```
.
├── README.md             # 本文件
├── notes/                # 文献精读笔记
└── (src / experiments 将随阶段推进添加)
```

## 📫 Author

- **杜可正 (Du Kezheng)** — 中国传媒大学 信息与通信工程学院
- 📧 202311103060@mails.cuc.edu.cn
- 🌐 [GitHub Profile](https://github.com/dook050915)

---

<sub>🚧 This is an active learning project. Code quality and structure will evolve. Issues and PRs welcome.</sub>
