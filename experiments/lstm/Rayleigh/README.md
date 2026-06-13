# LSTM 语义通信 · 瑞利衰落信道(平坦 Rayleigh + 完美 CSI 均衡)

## 实验目的

AWGN 只有加性噪声,是理想化的有线/视距场景。真实无线环境里信号经多条路径反射、不同时到达接收端,多径叠加使接收信号的幅度和相位随机起伏——这就是衰落。本实验把信道换成平坦瑞利衰落 y = h·x + n,在更现实的信道假设下检验同一套 LSTM 语义通信链路,与 AWGN 做同配置对比。

注意区分两个退化源:h 是**乘性衰落**(不是噪声),n 是加性噪声,瑞利信道下两者同时存在。

「平坦」意味着假设多径时延扩展远小于符号周期,所有路径叠成单个系数 h;若时延不可忽略则是频率选择性衰落、产生符号间干扰(ISI),本实验不建模那种情况。

## 数据集

数据来自 Europarl English,采样并过滤得到 50,000 条英文句子(保留 token 数 4–30 的句子,去除 Europarl 标签行、空行与括号形式的舞台说明)。与 real_channel 各实验同源、同切分(seed=42),保证可比。

## 信道模型

- 发送向量相邻两维 (I, Q) 凑成复符号:无线信号的等效基带表示天然是复数,多径叠加既缩放幅度也旋转相位,实数增益表达不了旋转
- h ~ CN(0, 1):实部虚部各自 N(0, ½),E[|h|²] = 1(衰落平均不改变信号功率,SNR 定义干净)。**|h| 的包络服从瑞利分布、相位均匀分布,信道因此得名**
- 每符号独立取 h(符号级平坦衰落,无时间相关性)
- 接收端假设完美 CSI,做迫零均衡:ŷ = y/h = x + n/h

均衡把衰落本身除掉了,但代价转嫁到噪声上:等效噪声功率变为 N₀/|h|²。瞬时 SNR = |h|²·平均SNR 服从指数分布,**任何平均 SNR 下深衰落(|h|→0)的概率都不为零**——这是 Rayleigh 性能差于 AWGN 的根本原因,也解释了为什么高 SNR 下差距依然存在(见下文结论)。

## 实现要点(code/channel.py · RayleighChannel)

- 输入 [B, n] 实信号(n 为偶数),`view_as_complex` 前需 `.contiguous()`
- 信号功率统计用 `.detach()`(噪声功率是环境量,梯度不应穿过功率估计;更早写的 AWGNChannel 未 detach,影响分析见 [experiments/README.md](../../README.md) 的「信道实现细节」)
- 其余链路(LSTM、channel encoder/decoder、功率归一化)与 AWGN 真信道版完全一致,唯一变量是信道层,保证两条曲线可直接对比

## 训练设置

```text
数据:Europarl English 50k(同 real_channel)
embedding_dim = 128
hidden_dim = 512
num_layers = 1
channel_dim = 256(沿用 AWGN channel_dim 消融的最优值,不重复消融)
batch_size = 96
learning_rate = 1e-3
epochs = 20,best checkpoint = epoch 8(Val Loss 3.4872,之后过拟合)
训练 SNR:每 batch 从 [-10, -5, 0, 5, 10, 15, 20] 随机取;验证固定 10 dB
BLEU:sacrebleu(corpus 级,tokenize=none),取值 0–1
```

## 结果

结果文件见 [multi_snr_50k_h512_c256/](./multi_snr_50k_h512_c256/)。

```text
SNR(dB) | Test Loss | BLEU
-10     | 3.9703    | 0.0382
-5      | 3.7770    | 0.0664
0       | 3.6095    | 0.1009
5       | 3.5165    | 0.1190
10      | 3.4759    | 0.1285
15      | 3.4598    | 0.1309
20      | 3.4533    | 0.1323
```

AWGN vs Rayleigh 对比(同 channel_dim=256):

![AWGN vs Rayleigh:左 Test Loss、右 BLEU 随 SNR 变化](../snr_sweep_curve.png)

## 结论

- **Rayleigh 全 SNR 段低于 AWGN**(20 dB 时 BLEU 0.13 vs 0.17)。瞬时 SNR = |h|²·平均SNR 服从指数分布,任何平均 SNR 下深衰落概率都不为零(20 dB 时仍有约 1% 的符号瞬时 SNR 低于 0 dB),这部分符号上的信息任何译码器都无法恢复——是测试时的固有信息损失,不是训练失败。
- 这与传统通信的经典结论一致:BPSK 误码率在 AWGN 下随 SNR 指数下降,在瑞利下只随 SNR 倒数(≈1/(4·SNR))下降。AWGN 曲线 10 dB 后真饱和(顶到模型自身天花板),Rayleigh 则以 1/SNR 的速度缓慢爬升,在 −10~20 dB 窗口内始终差着深衰落概率决定的缺口。BLEU-SNR 对比图是该经典结论的语义通信版本。
- 曲线仍随 SNR 单调上升,说明模型在衰落信道下同样学到了「信道质量↑ → 语义恢复↑」的映射,链路有效。

## 关于绝对 BLEU 偏低

本系统把整句压成一个 channel_dim 维定长向量过信道,信道使用量与句长无关;DeepSC 是逐 token 传输,信道使用量随句长线性增长。两者传输速率不在一个量级,绝对 BLEU 不可比。本实验的定位是「LSTM 语义通信 baseline 在 AWGN/Rayleigh 下的鲁棒性分析」,价值在曲线形态与对比,不在绝对值。

## 下一步

- 整句单向量 → 逐 token 传输:**已完成**,见 [tokenSeq2Seq/](./tokenSeq2Seq/)(瑞利下逐 token 同样胜整句)
- 编解码升级为 Transformer(DeepSC),复用同一条信道链(AWGN / Rayleigh 都已就位),做 LSTM vs Transformer 对比
