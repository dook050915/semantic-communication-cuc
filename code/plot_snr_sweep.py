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
    results1_path = "experiments/lstm/awgn/hidden_only/fixed_snr10_50k_h512/snr_sweep_results.txt"
    multi_results_path = "experiments/lstm/awgn/hidden_only/multi_snr_50k_h512/snr_sweep_results.txt"
    multi_hidden_cell_results_path = "experiments/lstm/awgn/hidden_cell/multi_snr_50k_h512/snr_sweep_results.txt"
    save_path = "experiments/lstm/awgn/hidden_cell/multi_snr_50k_h512/snr_sweep_curve.png"
    snrs, test_losses, bleus = load_snr_results(results1_path)
    multi_snrs, multi_test_losses, multi_bleus = load_snr_results(multi_results_path)
    multi_hidden_cell_snrs, multi_hidden_cell_test_losses, multi_hidden_cell_bleus = load_snr_results(multi_hidden_cell_results_path)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    axs[0].plot(snrs, test_losses, marker="o", label="Fixed-SNR, hidden only")
    axs[0].plot(multi_snrs, multi_test_losses, marker="x", color="red", label="Multi-SNR, hidden only")
    axs[0].plot(multi_hidden_cell_snrs, multi_hidden_cell_test_losses, marker="+", color="green", label="Multi-SNR, hidden + cell")

    axs[1].plot(snrs, bleus, marker="o", label="Fixed-SNR, hidden only")
    axs[1].plot(multi_snrs, multi_bleus, marker="x", color="red", label="Multi-SNR, hidden only")
    axs[1].plot(multi_hidden_cell_snrs, multi_hidden_cell_bleus, marker="+", color="green", label="Multi-SNR, hidden + cell")

    axs[0].legend()
    axs[1].legend()

    fig.suptitle("AWGN Robustness Comparison")
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
