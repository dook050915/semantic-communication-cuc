# 实验目录索引

本目录按「模型 → 信道 → 信道实现方式 → 实验配置」组织实验结果。

```text
experiments/
  lstm/
    snr_sweep_curve.png            # AWGN vs Rayleigh 对比图(同 channel_dim=256)
    noiseless/                     # 无信道 baseline
      20k_h256/
      50k_h256/
      50k_h512/
    awgn/
      bleu_snr_sacrebleu.png       # 三组 latent-noise 实验的 sacrebleu 对比曲线
      hidden_only/                 # latent-noise:仅对 hidden 加噪(早期对照)
        fixed_snr10_50k_h512/
        multi_snr_50k_h512/
      hidden_cell/                 # latent-noise:hidden + cell 同时加噪
        multi_snr_50k_h512/
      real_channel/                # 真信道:channel encoder/decoder + 功率归一化
        snr_sweep_curve.png        # channel_dim 消融对比图
        multi_snr_50k_h512_c32/
        multi_snr_50k_h512_c64/
        multi_snr_50k_h512_c128/
        multi_snr_50k_h512_c256/
        multi_snr_50k_h512_c512/
    Rayleigh/                      # 瑞利衰落真信道(命名说明见下)
      multi_snr_50k_h512_c256/
```

## 命名规则

- 第一层:模型结构(`lstm`,后续扩展 `transformer`)
- 第二层:信道条件(`noiseless` / `awgn` / `Rayleigh`)
- 第三层:信道实现方式(`hidden_only` / `hidden_cell` 直接对内部状态加噪;`real_channel` 为信道编解码 + 功率归一化的真信道)
- 最后一层:实验配置(如 `multi_snr_50k_h512`;真信道带 channel_dim 后缀,如 `_c256`)

两个历史遗留的不一致,为不破坏已有结果路径保持现状:`Rayleigh` 首字母大写而 `awgn` 小写;Rayleigh 只做了真信道实现,因此直接挂在第二层、省略 `real_channel` 中间层。

## 每个实验目录的文件

- `README.md`:实验目的、设置、结果和结论。真信道实验(`real_channel/` 的 5 组 channel_dim 消融、`Rayleigh/`)由父目录 README 统一说明,子目录只存结果文件
- `snr_sweep_results.txt`:SNR 扫描指标;`snr_sweep_curve.png`:对应曲线
- `train_log.txt`:训练日志;`vocab.json`:该实验词表
- 部分目录另有 `results.json`(单点评估)与 `prediction_samples.txt`(重构样例)

checkpoint(.pt)用于本地复现和继续评估,不上传 GitHub。

## BLEU 口径

BLEU 取值 0–1。早期实验(noiseless / hidden_only / hidden_cell)用自写 corpus 级 BLEU 计算;之后用 sacrebleu(tokenize=none)逐组校准,两者数值基本一致(各目录 `sacrebleu_results.txt`,对比曲线 `awgn/bleu_snr_sacrebleu.png`)。从 real_channel 起直接采用 sacrebleu。

## 信道实现细节:AWGN 的 signal_power 未 detach

`RayleighChannel` 对信号功率估计做了 `.detach()`——噪声功率是环境量,不应有梯度流过;`AWGNChannel` 写得更早,漏了这一步,事后发现(2026-06-11),影响面分析如下:

- **real_channel / Rayleigh 实验:不受影响。** 上游 `power_normalize` 把每样本功率钉死为常数,经 signal_power 的假梯度支路恒近零。
- **早期 latent-noise 实验(hidden_only / hidden_cell):支路有效,但结论不受影响。** 噪声功率与信号功率等比例绑定(SNR 恒定),模型无法借该支路改变相对噪声水平、不存在性能注水;它只引入一个被大张量均值(约 5 万元素)稀释的微弱尺度压力。且这批实验的结论均为同实现 A/B 对照(固定 vs 多 SNR、hidden vs hidden+cell),两臂瑕疵相同,相对结论成立。
- **处理:不回改 `AWGNChannel`**(改动梯度细节会破坏与已有实验线的严格可比性);后续新写信道代码时,环境量一律 detach。
