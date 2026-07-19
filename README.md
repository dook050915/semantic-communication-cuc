# Semantic Communication — CUC

文本语义通信系统(DeepSC 方向):用 PyTorch 从零实现文本语义通信的端到端系统,把语义表示作为受功率约束的信号送过 AWGN / 瑞利衰落信道,系统分析语义恢复质量(BLEU)随信道质量(SNR)的变化。沿 Farsad(2018)→ DeepSC(2021)的文献线做了 **LSTM** 与 **Transformer** 两代编解码器,在**同一条信道链路、同一评估口径**下对比。

状态:LSTM 主线全部完成(无信道 baseline → AWGN 状态加噪 → 真信道 + 功率归一化 + channel_dim 消融 → 瑞利衰落);逐 token + attention 版把绝对 BLEU 从 ~0.17 提到 ~0.87;Transformer 版完成 50k 同口径架构对比——Transformer 仅在 −10 dB 更高,LSTM 在 −5 dB 及以上明显领先。另记录了实际 110,844 句的探索性复跑,但它不是只改变数据量的单变量实验(见「核心结果」)。

## 核心结果

### 1. 逐 token LSTM:语义恢复性能随 SNR 降低逐步下降

Europarl 50k、hidden=512、channel_dim=256、多 SNR 随机训练,逐 token 传输 + 加性注意力,BLEU(sacrebleu,0–1):

| SNR (dB) | -10   | -5    | 0     | 5     | 10    | 15    | 20    |
| -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| AWGN     | 0.159 | 0.621 | 0.830 | 0.862 | 0.867 | 0.868 | 0.870 |
| Rayleigh | 0.076 | 0.162 | 0.365 | 0.615 | 0.753 | 0.802 | 0.816 |

- 两条曲线随 SNR 单调上升、高 SNR 饱和:端到端系统学到了「信道质量 ↑ → 语义恢复 ↑」。
- 瑞利全 SNR 段低于 AWGN:完美 CSI 均衡除掉了衰落,但深衰落符号上的噪声被 1/|h|² 放大,造成不可恢复的信息损失——与传统通信「瑞利差于 AWGN」的经典结论一致,可视为该结论的语义通信版本。
- channel_dim 存在速率-鲁棒性甜点(约 256),并非维度越大越好。

### 2. LSTM vs Transformer:Transformer 仅在 −10 dB 更高

主比较固定 Europarl 50k、AWGN、channel_dim=256,用逐 token LSTM(hidden=512)对比 Transformer(d_model=256、3 层、8 头)。表中同时保留 110k 探索性复跑用于审计,但不用于判断“增加数据量是否提升性能”:

| SNR (dB)                       | -10   | -5    | 0     | 10    | 20    |
| ------------------------------ | ----- | ----- | ----- | ----- | ----- |
| LSTM (50k)                     | 0.159 | 0.621 | 0.830 | 0.867 | 0.870 |
| Transformer d256 (50k)         | 0.283 | 0.443 | 0.508 | 0.538 | 0.539 |
| Transformer d256 (110k,探索性) | 0.279 | 0.390 | 0.431 | 0.450 | 0.452 |

- **严格同口径的 50k 对比**:−10 dB 时 Transformer 更高(0.283 vs 0.159);从 −5 dB 开始 LSTM 明显领先(−5 dB 为 0.621 vs 0.443,20 dB 为 0.870 vs 0.539)。
- **曲线形态**:Transformer 50k 的 BLEU 从 −10 到 20 dB 仅由 0.283 升至 0.539,LSTM 则由 0.159 升至 0.870。较平的 Transformer 曲线可能与解码器语言先验、信道表征利用不足或训练配置有关;这是待消融验证的解释,不是当前实验已经证明的根因。
- **110k 复跑不是数据量消融**:它同时改变了词表(16,471→25,826)、dropout(0→0.1)、训练轮数(40→30)和 train/val/test 划分。该组最优 val loss 出现在最后一轮,训练收敛也尚未闭合。因此不能由 0.452 的高 SNR BLEU 推出“更多数据无效”或“更多数据使性能下降”。
- **no-channel 仅是 sanity check**:train 0.21 / val 1.26 表明模型在无信道条件下能够优化、但泛化较弱;它不能单独排除其他实现或训练问题。

> 结论:当前能够直接支持的是 50k 同口径结果——Transformer 仅在 −10 dB 高于逐 token LSTM,LSTM 在 −5 dB 及以上明显领先。Transformer 为何没有随 SNR 提升而获得同等增益仍是开放问题;语言先验与信道利用不足是候选解释。110k 复跑用于记录探索过程,不作为数据规模效应的因果证据。

## 系统模型

```
源文本 s
  → Embedding (+ 位置编码, Transformer 版)
  → 语义编码器:LSTM / Transformer Encoder        → 每 token 语义表示
  → Channel Encoder (FC)                          → 发送符号 x
  → 功率归一化(仅真实 token 位置)→ 物理信道       → AWGN: y = x + n / Rayleigh: y = h·x + n(完美 CSI 均衡)
  → Channel Decoder (FC)                          → 恢复语义表示
  → 语义解码器:LSTM(加性注意力)/ Transformer Decoder + Linear  → 重建文本 ŝ

评估:BLEU(s, ŝ) 随 SNR ∈ [-10, 20] dB 变化(sacrebleu, corpus 级)
```

LSTM 与 Transformer **复用同一条信道链**(channel encoder/decoder + 功率归一化 + AWGN/Rayleigh),只替换语义编解码器,保证对比可归因于架构。

## 实验

数据:Europarl 英文语料,过滤保留 4–30 词的句子,采样 50k(及 20k 对照)。train / val / test = 8 / 1 / 1,固定随机种子。

- `experiments/lstm/noiseless/` — 无信道 baseline,对照数据量与模型容量
- `experiments/lstm/awgn/{hidden_only,hidden_cell}/` — 早期状态加噪敏感性分析(固定 vs 多 SNR;hidden vs hidden+cell)
- `experiments/lstm/awgn/real_channel/` — 真信道 + 功率归一化,channel_dim 消融(16–512),整句版与逐 token 版
- `experiments/lstm/Rayleigh/` — 平坦瑞利 + 完美 CSI 均衡,与 AWGN 同配置对比
- `experiments/transformer/{AWGN,Rayleigh}/` — Transformer 版 AWGN/Rayleigh 实验

每组实验的设置、结果、失败样例与结论见对应目录的 README。

## 踩过的坑

记录是为了说明结果为何可信,也是这个项目最花时间的部分。

**信道与训练**

1. **功率归一化的 PAD 污染**:`sequence_power_normalize` 早期把 PAD 位置算进功率均值,使每句获得 0~+7 dB 不等的隐性 SNR 增益,定量结果虚高。修复:功率均值只在真实 token 位置上求(乘 mask 后再平均)。修复前的旧数据已作废并在目录 README 注明。
2. **功率归一化必须可微**:不能放进 `no_grad`,否则梯度断在信道处、encoder 收不到回传。信道里的环境量(signal_power 等)则要 `detach`,避免假梯度支路;AWGN 早期未 detach 的影响面已单独分析(噪声与信号等比例,模型无法借此改变 SNR,相对结论不受影响)。
3. **瑞利逐 token 的 SNR 定义陷阱**:实维功率=1 时复符号能量=2,N0 取 1 还是 2 差 3 dB。逐 token 分支按此对齐口径,否则与整句版/AWGN 不可比。
4. **cuDNN 要求 LSTM 隐状态连续**:`transmit` 里 permute 后必须 `.contiguous()`,CPU/MPS 宽容、GPU 才报 `hx is not contiguous`。

**Transformer**

5. **两套相反的 mask 约定**:PyTorch 的 `key_padding_mask` 是 **True=屏蔽**,而信道的功率归一化用 **True=有效**。在 Seq2Seq 层用 `mask` 喂信道、`~mask` 喂解码器,传混会导致注意力关注 PAD、loss 不收敛。
6. **训练 recipe(Transformer 比 LSTM 吃这个)**:固定 lr=1e-4 无 warmup → 欠训练,val 卡在高位;`1e-3` 无 warmup 会被开头的大梯度带崩。正确做法:**warmup + cosine 衰减 + 峰值 5e-4 + grad_clip 1.0**,scheduler 按 step 调度。
7. **d_model 与 channel_dim 的秩瓶颈**:当 d_model < channel_dim,channel encoder 把低维表示「扩」成更高维信道符号,秩 ≤ d_model,多出的维度只挨噪声、不携带信息,白白浪费一半信道。须 **d_model ≥ channel_dim**,或采用 DeepSC 式压缩(channel_dim ≤ d_model)。
8. **MPS 不支持 nested_tensor**:`nn.TransformerEncoder` 的提速路径在 Mac MPS 上报 `_nested_tensor_from_mask_left_aligned`,设 `enable_nested_tensor=False` 关掉(纯速度优化,不影响结果)。

**架构对比的判断**

9. **深层 LSTM 难训**:相同训练设置下,3 层 LSTM 训不动/不稳(AWGN 卡在高 loss,Rayleigh 收敛但不如 1 层)——深 RNN 优化困难,堆深度无增益。故主对比采用各架构各自合理的配置,并如实记录。
10. **no-channel 对照是 sanity check,不是排错证明**:无信道自编码结果显示模型能够降低训练损失,同时存在明显 train/val gap。这帮助区分基础优化能力与信道影响,但不能单独排除 mask、解码策略或训练配置等问题。

## 局限

- 本仓库是「LSTM/Transformer 语义通信 baseline + 信道鲁棒性分析 + 架构对比」,不是完整 DeepSC 复现:未加互信息损失项;更大数据的 Transformer 复跑也不是严格控制变量实验,且可能尚未充分收敛。
- 词向量从零训练,长句、低频词与专有名词的重构较弱(各实验 README 附失败样例)。
- 50k 数据对 Transformer 偏小;110k 探索性复跑同时改变了词表、dropout、训练轮数与数据划分,因此「仅增加训练数据后结论是否改变」仍未回答。

## 复现

环境:Python 3.11,`pip install -r requirements.txt`。

```bash
# 1. 数据:下载 Europarl v7 英文语料(https://www.statmt.org/europarl/),
#    解压到 data/raw/europarl-v7/txt/en/,然后过滤采样:
python code/prepare_data.py

# 2. 训练:配置集中在 train_lstm.py / train_transformer.py 顶部的 config
python code/train_lstm.py          # LSTM
python code/train_transformer.py   # Transformer(warmup + cosine + grad_clip)

# 3. 评估:对 best checkpoint 做 SNR 扫描,输出各 SNR 的 Test Loss 与 BLEU
python code/sweep.py
```

BLEU 用 sacrebleu(corpus 级,tokenize=none),取值 0–1。

## 路线

| 阶段                     | 内容                                                                | 状态         |
| ------------------------ | ------------------------------------------------------------------- | ------------ |
| 1. LSTM baseline(无信道) | Seq2Seq 自编码,端到端重建,BLEU 评估                                 | 完成         |
| 2. LSTM + AWGN 信道      | 状态加噪分析 → 真信道 + 功率归一化;SNR 扫描、channel_dim 消融       | 完成         |
| 3. 瑞利衰落信道          | 平坦瑞利 + 完美 CSI 均衡,与 AWGN 同配置对比                         | 完成         |
| 4. 逐 token + attention  | 整句压缩 → 逐 token 传输,BLEU ~0.17 → ~0.87                         | 完成         |
| 5. Transformer 对比      | 保持 AWGN 信道与 channel_dim=256,仅替换编解码器,LSTM vs Transformer | 完成         |
| 6. 更大数据探索性复跑    | 实际 110,844 句 + dropout 0.1;未启用 label smoothing,且非单变量对照 | 完成(探索性) |

## 未来工作

当前结果已能支撑「LSTM/Transformer 在小数据下的对比」这一结论,以下是把项目继续做深的方向:

- **句子切分扩容数据**:当前预处理把每行当一句,直接丢弃 >30 词的长行(占内容行约 83%)。在长度过滤前引入句子切分(如 NLTK `sent_tokenize`),可把可用语料从约 11 万句扩到约 150 万句,系统检验数据规模对 Transformer 过拟合的影响。
- **互信息损失(MI loss)**:DeepSC 的核心配方之一,本仓库未实现。加入 MI 损失项有望提升信道编码的传输效率。
- **严格的数据规模消融**:先从完整语料中固定一份公共 test set 与固定词表/子词 tokenizer,再从剩余训练集抽取 50k 与 110k 子集;保持 dropout、训练步数口径、学习率策略和随机种子一致,只改变训练样本量。
- **语义层评估**:除 BLEU 外补充句向量相似度(如 Sentence-BERT),更贴合「语义」恢复的本意。

## 关键文献

1. Xie, Qin, Li, Juang. _Deep Learning Enabled Semantic Communication Systems_. IEEE TSP 2021.(DeepSC,[精读笔记](notes/deepsc-2021.md))
2. Farsad, Rao, Goldsmith. _Deep Learning for Joint Source-Channel Coding of Text_. ICASSP 2018.
3. O'Shea, Hoydis. _An Introduction to Deep Learning for the Physical Layer_. IEEE TCCN 2017.([精读笔记](notes/oshea-2017.md))

## 代码结构

```
.
├── README.md
├── requirements.txt
├── code/            数据、模型(LSTM / Transformer)、训练、评估、信道
└── experiments/     实验结果,按「模型 / 信道 / 信道实现方式 / 配置」组织
```

技术栈:Python 3.11 · PyTorch · sacrebleu · matplotlib

## Author

杜可正 — 中国传媒大学 信息与通信工程学院
202311103060@mails.cuc.edu.cn
