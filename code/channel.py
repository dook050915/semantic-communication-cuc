import torch
import torch.nn as nn

class AWGNChannel(nn.Module):
    """
    AWGN 信道层。

    输入:
        x: 任意形状的连续张量，例如 hidden [num_layers, batch_size, hidden_dim]
        snr_db: 信噪比，单位 dB

    输出:
        noisy_x: 和 x 形状相同的张量
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, snr_db):
        """
        对输入张量 x 加 AWGN 噪声。

        shape:
            x: 任意 shape
            noisy_x: 和 x 完全相同 shape
        """
        if snr_db is None:
            return x

        signal_power = x.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = noise_power.sqrt()
        
        noise = torch.randn_like(x) * noise_std
        noisy_x = x + noise
        return noisy_x

channel_types = {
    "AWGN": AWGNChannel,
}