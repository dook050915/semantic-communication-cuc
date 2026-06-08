from prepare_data import resolve_path
import matplotlib.pyplot as plt

def load_snr_results(path):
    snrs,test_losses,bleus = [],[],[]
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        i=0
        for line in f:
            if i==0 or line.strip()=="":
                i+=1
                continue
            tokens = [token.strip() for token in line.split("|")]
            snrs.append(float(tokens[0]))
            test_losses.append(float(tokens[1]))
            bleus.append(float(tokens[2]))
    return snrs,test_losses,bleus

def plot_snr_curves(snrs, losses, bleus, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(snrs, losses, marker="o")
    axs[1].plot(snrs, bleus, marker="o")

    fig.suptitle("SNR Sweep Curves")
    axs[0].set_xlabel("SNR(dB)")
    axs[0].set_ylabel("Test Loss")
    axs[1].set_xlabel("SNR(dB)")
    axs[1].set_ylabel("BLEU")
    axs[0].grid()
    axs[1].grid()
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()



def main():
    results_path = "experiments/lstm_awgn_50k_h512_snr10/snr_sweep_results.txt"
    save_path = "experiments/lstm_awgn_50k_h512_snr10/snr_sweep_curve.png"
    snrs,test_losses,bleus = load_snr_results(results_path)
    plot_snr_curves(snrs,test_losses,bleus, resolve_path(save_path))


if __name__ == "__main__":
    main()
