from pathlib import Path

import matplotlib.pyplot as plt

from prepare_data import resolve_path


OLD_TOKEN_C256_A128 = {
    "snr": [-10, -5, 0, 5, 10, 15, 20],
    "loss": [2.2212, 0.4639, 0.1528, 0.0998, 0.0896, 0.0855, 0.0851],
    "bleu": [0.2659, 0.6979, 0.8497, 0.8731, 0.8779, 0.8790, 0.8799],
}


def load_snr_results(path):
    snrs, test_losses, bleus = [], [], []
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no == 0 or line.strip() == "":
                continue
            tokens = [token.strip() for token in line.split("|")]
            snrs.append(float(tokens[0]))
            test_losses.append(float(tokens[1]))
            bleus.append(float(tokens[2]))
    return {"snr": snrs, "loss": test_losses, "bleu": bleus}


def plot_loss_bleu(series, title, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    for item in series:
        data = item["data"]
        axs[0].plot(
            data["snr"],
            data["loss"],
            marker=item.get("marker", "o"),
            label=item["label"],
        )
        axs[1].plot(
            data["snr"],
            data["bleu"],
            marker=item.get("marker", "o"),
            label=item["label"],
        )

    axs[0].set_title("Test Loss")
    axs[1].set_title("BLEU")
    axs[0].set_xlabel("SNR (dB)")
    axs[1].set_xlabel("SNR (dB)")
    axs[0].set_ylabel("Test Loss")
    axs[1].set_ylabel("BLEU")

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    base = Path("experiments/lstm/awgn/real_channel")
    token_base = base / "tokenSeq2Seq"
    state_base = base / "stateSeq2Seq"

    token_c16_a128 = load_snr_results(
        token_base / "multi_snr_50k_h512_c16_a128" / "snr_sweep_results.txt"
    )
    token_c128_a128 = load_snr_results(
        token_base / "multi_snr_50k_h512_c128_a128" / "snr_sweep_results.txt"
    )
    token_c256_a128 = load_snr_results(
        token_base / "multi_snr_50k_h512_c256_a128" / "snr_sweep_results.txt"
    )
    state_c256 = load_snr_results(
        state_base / "multi_snr_50k_h512_c256" / "snr_sweep_results.txt"
    )

    plot_loss_bleu(
        [
            {
                "label": "c256_a128 v1 deprecated",
                "data": OLD_TOKEN_C256_A128,
                "marker": "x",
            },
            {
                "label": "c256_a128 masked-power",
                "data": token_c256_a128,
                "marker": "o",
            },
        ],
        "TokenSeq2Seq c256_a128: Before vs After Power Mask Fix",
        token_base / "c256_a128_v1_vs_masked.png",
    )

    plot_loss_bleu(
        [
            {"label": "c16_a128", "data": token_c16_a128, "marker": "o"},
            {"label": "c128_a128", "data": token_c128_a128, "marker": "s"},
            {"label": "c256_a128", "data": token_c256_a128, "marker": "^"},
        ],
        "AWGN TokenSeq2Seq Channel Dimension Sweep",
        token_base / "token_channel_dim_sweep_a128.png",
    )

    plot_loss_bleu(
        [
            {"label": "TokenSeq2Seq c16_a128", "data": token_c16_a128, "marker": "o"},
            {"label": "StateSeq2Seq c256", "data": state_c256, "marker": "s"},
        ],
        "TokenSeq2Seq c16_a128 vs StateSeq2Seq c256",
        token_base / "token_c16_a128_vs_state_c256.png",
    )


if __name__ == "__main__":
    main()
