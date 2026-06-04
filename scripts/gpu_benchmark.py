import argparse
import time

import torch
import torch.nn as nn


def mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 ** 2:.1f} MB"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_ms(fn, device: torch.device, repeat: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    synchronize(device)
    end = time.perf_counter()
    return (end - start) * 1000 / repeat


def benchmark_matmul(device: torch.device, n: int, repeat: int) -> None:
    print("\n=== Matrix multiplication ===")
    print(f"Device: {device}")
    print(f"Shape: {n} x {n}")

    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    elapsed_ms = time_ms(lambda: a @ b, device=device, repeat=repeat, warmup=5)
    tflops = (2 * n**3) / (elapsed_ms / 1000) / 1e12

    print(f"Average time: {elapsed_ms:.2f} ms")
    print(f"Approx throughput: {tflops:.2f} TFLOPS")

    if device.type == "cuda":
        print(f"CUDA memory allocated: {mb(torch.cuda.memory_allocated())}")


class TinyLSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        out, _ = self.encoder(x)
        return self.classifier(out)


def benchmark_lstm(device: torch.device, repeat: int) -> None:
    print("\n=== LSTM training step ===")
    print("This is closer to the DeepSC LSTM baseline than an image CNN benchmark.")

    vocab_size = 8000
    batch_size = 96
    seq_len = 32
    embed_dim = 256
    hidden_dim = 512
    layers = 2

    model = TinyLSTMModel(vocab_size, embed_dim, hidden_dim, layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss

    last_loss = None
    for _ in range(3):
        last_loss = step()
    synchronize(device)

    elapsed_ms = time_ms(step, device=device, repeat=repeat, warmup=5)
    tokens_per_sec = batch_size * seq_len / (elapsed_ms / 1000)

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Average train step: {elapsed_ms:.2f} ms")
    print(f"Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"Last loss: {last_loss.item():.4f}")

    if device.type == "cuda":
        print(f"Peak CUDA memory allocated: {mb(torch.cuda.max_memory_allocated())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch GPU benchmark for the Windows RTX 4060 machine.")
    parser.add_argument("--matmul-size", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of CUDA.")
    args = parser.parse_args()

    print("=== PyTorch benchmark ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Activate the CUDA PyTorch environment on Windows.")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"Total VRAM: {mb(props.total_memory)}")
        print(f"Compute capability: {props.major}.{props.minor}")

    benchmark_matmul(device, n=args.matmul_size, repeat=args.repeat)
    benchmark_lstm(device, repeat=args.repeat)
    print("\nDone.")


if __name__ == "__main__":
    main()
