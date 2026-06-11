# 瑞利衰落信道实验

平坦瑞利衰落 + 完美 CSI 均衡(y = h·x + n,复符号),与 AWGN 真信道同配置(channel_dim=256)对比。链路其余部分与 `awgn/real_channel/` 完全一致,唯一变量是信道层。

![AWGN vs Rayleigh:左 Test Loss、右 BLEU 随 SNR 变化](../snr_sweep_curve.png)

结论:Rayleigh 全 SNR 段低于 AWGN(20 dB 时 BLEU 0.13 vs 0.17)。均衡把衰落除掉,但深衰落符号上的噪声被 1/|h|² 放大——瞬时 SNR 服从指数分布,任何平均 SNR 下深衰落概率都不为零,这部分信息损失任何译码器都无法恢复。缺口大小及随 SNR 的变化规律与传统通信中瑞利信道差于 AWGN 的经典结论一致;曲线仍随 SNR 单调上升,链路在衰落信道下有效。

信道模型、实现要点、训练设置与完整结果分析见 [multi_snr_50k_h512_c256/](./multi_snr_50k_h512_c256/)。
