# 实验目录索引

本目录按“模型 -> 信道 -> 状态扰动方式 -> 实验配置”组织实验结果。

当前结构：

```text
experiments/
  lstm/
    noiseless/
      20k_h256/
      50k_h256/
      50k_h512/
    awgn/
      hidden_only/
        fixed_snr10_50k_h512/
        multi_snr_50k_h512/
      hidden_cell/
        multi_snr_50k_h512/
```

## 命名规则

- 第一层是模型结构，例如 `lstm`，后续可以扩展为 `transformer`。
- 第二层是信道条件，例如 `noiseless` 或 `awgn`。
- 第三层是状态扰动方式，例如 `hidden_only` 或 `hidden_cell`。
- 最后一层是具体实验配置，例如 `multi_snr_50k_h512`。

每个具体实验目录保存该实验产生的结果文件，例如：

- `README.md`：实验目的、设置、结果和结论
- `results.json`：单点评估结果
- `prediction_samples.txt`：重构样例
- `snr_sweep_results.txt`：SNR sweep 指标
- `snr_sweep_curve.png`：SNR sweep 曲线
- `train_log.txt`：训练日志
- `vocab.json`：该实验对应词表

checkpoint 文件用于本地复现和继续评估，不作为 GitHub 展示文件。
