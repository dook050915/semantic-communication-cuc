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
    results_awgn_path = "experiments/lstm/awgn/real_channel/multi_snr_50k_h512_c256/snr_sweep_results.txt"
    results_rayleigh_path = "experiments/lstm/rayleigh/multi_snr_50k_h512_c256/snr_sweep_results.txt"
    

    save_path = "experiments/lstm/snr_sweep_curve.png"
    snrs_awgn, test_losses_awgn, bleus_awgn = load_snr_results(results_awgn_path)
    snrs_rayleigh, test_losses_rayleigh, bleus_rayleigh = load_snr_results(results_rayleigh_path)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    axs[0].plot(snrs_awgn, test_losses_awgn, marker="o", label="AWGN")
    axs[0].plot(snrs_rayleigh, test_losses_rayleigh, marker="x", color="red", label="Rayleigh")
    
    axs[1].plot(snrs_awgn, bleus_awgn, marker="o", label="AWGN")
    axs[1].plot(snrs_rayleigh, bleus_rayleigh, marker="x", color="red", label="Rayleigh")

    axs[0].legend()
    axs[1].legend()

    fig.suptitle("AWGN vs Rayleigh(Channel Dim: 256)")
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
