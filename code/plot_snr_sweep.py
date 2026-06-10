from prepare_data import resolve_path
import matplotlib.pyplot as plt

def load_snr_results(path):
    snrs, test_losses, bleus = [], [], []
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        i = 0
        for line in f:
            if i == 0 or line.strip() == "":
                i += 1
                continue
            tokens = [token.strip() for token in line.split("|")]
            snrs.append(float(tokens[0]))
            test_losses.append(float(tokens[1]))
            bleus.append(float(tokens[2]))
    return snrs,test_losses,bleus


def main():
    results_c32_path = "experiments/lstm/awgn/real_channel/multi_snr_50k_h512_c32/snr_sweep_results.txt"
    results_c64_path = "experiments/lstm/awgn/real_channel/multi_snr_50k_h512_c64/snr_sweep_results.txt"
    results_c128_path = "experiments/lstm/awgn/real_channel/multi_snr_50k_h512_c128/snr_sweep_results.txt"
    results_c256_path = "experiments/lstm/awgn/real_channel/multi_snr_50k_h512_c256/snr_sweep_results.txt"
    results_c512_path = "experiments/lstm/awgn/real_channel/multi_snr_50k_h512_c512/snr_sweep_results.txt"

    save_path = "experiments/lstm/awgn/real_channel/snr_sweep_curve.png"
    snrs_c32, test_losses_c32, bleus_c32 = load_snr_results(results_c32_path)
    snrs_c64, test_losses_c64, bleus_c64 = load_snr_results(results_c64_path)
    snrs_c128, test_losses_c128, bleus_c128 = load_snr_results(results_c128_path)
    snrs_c256, test_losses_c256, bleus_c256 = load_snr_results(results_c256_path)
    snrs_c512, test_losses_c512, bleus_c512 = load_snr_results(results_c512_path)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    axs[0].plot(snrs_c32, test_losses_c32, marker="o", label="Channel Dim: 32")
    axs[0].plot(snrs_c64, test_losses_c64, marker="x", color="red", label="Channel Dim: 64")
    axs[0].plot(snrs_c128, test_losses_c128, marker="o", color="green", label="Channel Dim: 128")
    axs[0].plot(snrs_c256, test_losses_c256, marker="x", color="blue", label="Channel Dim: 256")
    axs[0].plot(snrs_c512, test_losses_c512, marker="o", color="orange", label="Channel Dim: 512")

    axs[1].plot(snrs_c32, bleus_c32, marker="o", label="Channel Dim: 32")
    axs[1].plot(snrs_c64, bleus_c64, marker="x", color="red", label="Channel Dim: 64")
    axs[1].plot(snrs_c128, bleus_c128, marker="o", color="green", label="Channel Dim: 128")
    axs[1].plot(snrs_c256, bleus_c256, marker="x", color="blue", label="Channel Dim: 256")
    axs[1].plot(snrs_c512, bleus_c512, marker="o", color="orange", label="Channel Dim: 512")

    axs[0].legend()
    axs[1].legend()

    fig.suptitle("AWGN BLEU-SNR vs channel_dim")
    axs[0].set_xlabel("SNR(dB)")
    axs[0].set_ylabel("Test Loss")
    axs[1].set_xlabel("SNR(dB)")
    axs[1].set_ylabel("BLEU")
    axs[0].grid()
    axs[1].grid()
    fig.tight_layout()
    plt.savefig(resolve_path(save_path))
    plt.close()


if __name__ == "__main__":
    main()
