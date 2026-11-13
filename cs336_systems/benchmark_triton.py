import gc
from collections.abc import Callable

import torch
import triton
from torch import OutOfMemoryError, Tensor

from cs336_basics.attention import scaled_dot_product_attention
from cs336_systems.flash_attention_triton import FlashAttention2TritonFunc

device = "cuda"
batch_size = 1
seq_lens = [2 ** exponent for exponent in range(7, 17)]  # 128 ... 65536
d_models = [16, 32, 64, 128]
precisions = {"fp32": torch.float32, "bf16": torch.bfloat16}

# Causal attention over (Q, K, V), returning O.
Attention = Callable[[Tensor, Tensor, Tensor], Tensor]


def do_bench(fn: Callable[[], object], grad_to_none: list[Tensor] | None = None) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=grad_to_none)


def forward_only(attention: Attention, Q: Tensor, K: Tensor, V: Tensor, grad_out: Tensor) -> float:
    return do_bench(lambda: attention(Q, K, V))


def backward_only(attention: Attention, Q: Tensor, K: Tensor, V: Tensor, grad_out: Tensor) -> float:
    # One forward outside the timing; retain_graph keeps what it saved, so every timed call can run backward on it.
    O = attention(Q, K, V)
    return do_bench(lambda: O.backward(grad_out, retain_graph=True), grad_to_none=[Q, K, V])


def forward_backward(attention: Attention, Q: Tensor, K: Tensor, V: Tensor, grad_out: Tensor) -> float:
    return do_bench(lambda: attention(Q, K, V).backward(grad_out), grad_to_none=[Q, K, V])


MEASUREMENTS = {"forward": forward_only, "backward": backward_only, "forward+backward": forward_backward}


def benchmark(label: str, attention: Attention, Q: Tensor, K: Tensor, V: Tensor, grad_out: Tensor) -> dict[str, str]:
    results = {}
    for name, measure in MEASUREMENTS.items():
        try:
            results[f"{label} {name} (ms)"] = f"{measure(attention, Q, K, V, grad_out):.3f}"
        except OutOfMemoryError:
            results[f"{label} {name} (ms)"] = "OOM"
        gc.collect()
        torch.cuda.empty_cache()
    return results


def markdown_table(rows: list[dict]) -> str:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main():
    rows = []
    for precision, dtype in precisions.items():
        for d_model in d_models:
            for seq_len in seq_lens:
                torch._dynamo.reset()
                shape = (batch_size, seq_len, d_model)
                Q, K, V = (torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for _ in range(3))
                grad_out = torch.randn(shape, device=device, dtype=dtype)
                row = {"precision": precision, "d_model": d_model, "seq_len": seq_len}

                row |= benchmark("flash", lambda Q, K, V: FlashAttention2TritonFunc.apply(Q, K, V, True), Q, K, V, grad_out)

                causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
                row |= benchmark("regular", lambda Q, K, V: scaled_dot_product_attention(Q, K, V, causal_mask), Q, K, V, grad_out)
                del causal_mask

                rows.append(row)
                print(row, flush=True)  # progress, and partial results if the run is cut short

    print(markdown_table(rows))


if __name__ == "__main__":
    main()
