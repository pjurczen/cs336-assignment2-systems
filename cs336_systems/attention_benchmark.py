import statistics
import timeit
from collections.abc import Callable

import torch
from torch import OutOfMemoryError

from cs336_basics.attention import scaled_dot_product_attention

device = "cuda"
batch_size = 8
d_models = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096, 8192, 16384, 32768]


def synchronize(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda" or device == "cuda:0":
        torch.cuda.synchronize()


def memory_allocated(device: str) -> int:
    if device == "mps":
        return torch.mps.current_allocated_memory()
    elif device.startswith("cuda"):
        return torch.cuda.memory_allocated()


def empty_cache(device: str):
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def result_keys(label: str) -> list[str]:
    return [f"forward{label} (ms)", f"backward{label} (ms)", f"memory before backward{label} (MiB)", f"added by forward{label} (MiB)"]


def attention_benchmark(attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor], label: str, d_model: int,
                        seq_len: int) -> dict:
    casual_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    Q = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
    K = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
    V = torch.randn((batch_size, seq_len, d_model), device=device, requires_grad=True)
    grad_out = torch.randn((batch_size, seq_len, d_model), device=device)
    backward_timing = []
    forward_timing = []

    print(f"Benchmarking{label} d_model: {d_model}, seq_len: {seq_len}")

    # Warmup steps (for the compiled variant, the first forward and backward also compile)
    for _ in range(2):
        logits: torch.Tensor = attention(Q, K, V, casual_mask)
        logits.backward(grad_out)
    synchronize(device)
    before_forward_memory: int = 0
    after_forward_memory: int = 0
    for i in range(100):
        before_forward_memory = memory_allocated(device)

        t0 = timeit.default_timer()
        logits: torch.Tensor = attention(Q, K, V, casual_mask)
        synchronize(device)
        t1 = timeit.default_timer()

        after_forward_memory = memory_allocated(device)

        logits.backward(grad_out)
        synchronize(device)
        t2 = timeit.default_timer()
        forward_timing.append(t1 - t0)
        backward_timing.append(t2 - t1)

    forward_key, backward_key, memory_key, added_key = result_keys(label)
    return {
        forward_key: f"{statistics.mean(forward_timing) * 1000:.2f} ± {statistics.stdev(forward_timing) * 1000:.2f}",
        backward_key: f"{statistics.mean(backward_timing) * 1000:.2f} ± {statistics.stdev(backward_timing) * 1000:.2f}",
        memory_key: f"{after_forward_memory / 1024 ** 2:.2f}",
        added_key: f"{(after_forward_memory - before_forward_memory) / 1024 ** 2:.2f}",
    }


def markdown_table(rows: list[dict]) -> str:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


rows = []
for d_model in d_models:
    for seq_len in seq_lens:
        row = {"d_model": d_model, "seq_len": seq_len}
        # Fresh Dynamo state per shape: compile statically for this configuration, and stay clear of the recompile limit.
        torch._dynamo.reset()
        variants = {"": scaled_dot_product_attention, " compiled": torch.compile(scaled_dot_product_attention)}
        for label, attention in variants.items():
            try:
                row |= attention_benchmark(attention, label, d_model, seq_len)
            except OutOfMemoryError:
                print(f"OutOfMemoryError{label} d_model: {d_model}, seq_len: {seq_len}")
                row |= {key: "OOM" for key in result_keys(label)}
            empty_cache(device)
        rows.append(row)

print(markdown_table(rows))
