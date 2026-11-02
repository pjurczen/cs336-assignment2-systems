import argparse
import statistics
import timeit

import torch

from cs336_basics.adamw_optimizer import AdamWOptimizer
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.muon_optimizer import MuonOptimizer
from cs336_basics.transformer_lm import TransformerLM


def benchmark(args: argparse.Namespace):
    raw_model: torch.nn.Module = TransformerLM(
        vocab_size=args.model_vocab_size,
        context_length=args.model_context_length,
        num_layers=args.model_num_layers,
        d_model=args.model_d_model,
        num_heads=args.model_num_heads,
        d_ff=args.model_d_ff,
        eps=args.model_eps,
        theta=args.model_rope_theta,
        device=args.device,
        dtype=torch.float32,
        tie_embeddings=args.model_tie_embeddings,
        qk_norm=args.model_qk_norm,
        logit_soft_cap=args.model_logit_soft_cap,
    )
    if args.device.startswith("cuda"):
        # Let float32 matmuls run on TF32 tensor cores (A100 and newer).
        torch.set_float32_matmul_precision("high")

    # aot_eager works on MPS; on CUDA the default inductor backend generates fused kernels.
    compile_backend: str = "inductor" if args.device.startswith("cuda") else "aot_eager"
    # Forward passes go through the compiled wrapper; checkpoints use the plain model so keys have no "_orig_mod." prefix.
    model: torch.nn.Module = torch.compile(raw_model, backend=compile_backend) if args.model_compile else raw_model

    # With --optimizer_muon, the 2D matrices inside the Transformer blocks go to Muon; everything else stays on AdamW.
    matrix_params = []
    other_params = []
    for name, p in raw_model.named_parameters():
        if args.optimizer_muon and p.ndim == 2 and name.startswith("layers"):
            matrix_params.append(p)
        else:
            other_params.append(p)

    adamw: torch.optim.Optimizer = AdamWOptimizer(
        params=other_params,
        lr=args.optimizer_lr_min,
        weight_decay=args.optimizer_weight_decay,
        betas=(args.optimizer_beta1, args.optimizer_beta2),
        eps=args.optimizer_eps,
    )
    muon: torch.optim.Optimizer | None = MuonOptimizer(
        params=matrix_params,
        lr=args.optimizer_lr_min,
        weight_decay=args.optimizer_weight_decay,
    ) if args.optimizer_muon else None
    optimizers: list[torch.optim.Optimizer] = [adamw] if muon is None else [adamw, muon]

    inputs, targets = get_random_input_batch(batch_size=args.batch_size, context_length=args.model_context_length, vocab_size=args.model_vocab_size, device=args.device, dtype=torch.long)

    forward_timing = []
    backward_timing = []
    optimizers_timing = []

    for i in range(1, args.w + args.n + 1):

        t0 = timeit.default_timer()
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.bf16):
            logits = model(inputs)

        synchronize(args.device)
        t1 = timeit.default_timer()

        loss: torch.Tensor = cross_entropy(logits, targets)

        synchronize(args.device)
        t2 = timeit.default_timer()

        loss.backward()

        synchronize(args.device)
        t3 = timeit.default_timer()

        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        synchronize(args.device)
        t4 = timeit.default_timer()

        if i > args.w:
            forward_timing.append(t1 - t0)
            backward_timing.append(t3 - t2)
            optimizers_timing.append(t4 - t3)

        if i == args.w:
            print("Warmup done")

    print(f"Warmup iters: {args.w}")
    print(f"Measurement iters: {args.n}")

    print(f"Forward pass: {statistics.mean(forward_timing) * 1000:.2f} ± {statistics.stdev(forward_timing) * 1000:.2f} ms")
    print(f"Forward pass values: {forward_timing}")

    print(f"Backward pass: {statistics.mean(backward_timing) * 1000:.2f} ± {statistics.stdev(backward_timing) * 1000:.2f} ms")
    print(f"Backward pass values: {backward_timing}")

    print(f"Optimizer step: {statistics.mean(optimizers_timing) * 1000:.2f} ± {statistics.stdev(optimizers_timing) * 1000:.2f} ms")
    print(f"Optimizer step values: {optimizers_timing}")


def synchronize(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda" or device == "cuda:0":
        torch.cuda.synchronize()


def get_random_input_batch(batch_size: int, context_length: int, vocab_size: int, device: str, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    random_data = torch.randint(0, vocab_size, (batch_size, context_length + 1), dtype=dtype, device=device)
    inputs = random_data[..., : -1]
    targets = random_data[..., 1:]
    return inputs, targets


def int_or_none(value: str) -> int | None:
    # Lets a flag be switched off from the command line, e.g. --model_rope_theta none (no RoPE).
    return None if value.lower() == "none" else int(value)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--optimizer_lr_min", type=float, default=1e-4)
    arg_parser.add_argument("--optimizer_lr_max", type=float, default=1e-3)
    arg_parser.add_argument("--optimizer_warmup_iters", type=int, default=None)
    arg_parser.add_argument("--optimizer_cosine_cycle_iters", type=int, default=None)
    arg_parser.add_argument("--optimizer_weight_decay", type=float, default=0.01)
    arg_parser.add_argument("--optimizer_beta1", type=float, default=0.9)
    arg_parser.add_argument("--optimizer_beta2", type=float, default=0.999)
    arg_parser.add_argument("--optimizer_eps", type=float, default=1e-8)
    arg_parser.add_argument("--optimizer_muon", action="store_true",
                            help="Muon for the 2D matrices inside the blocks, AdamW for the rest")
    arg_parser.add_argument("--model_context_length", type=int, default=512)
    arg_parser.add_argument("--model_vocab_size", type=int, default=10000)
    arg_parser.add_argument("--model_num_layers", type=int, default=12)
    arg_parser.add_argument("--model_d_model", type=int, default=768)
    arg_parser.add_argument("--model_d_ff", type=int, default=3072)
    arg_parser.add_argument("--model_num_heads", type=int, default=12)
    arg_parser.add_argument("--model_eps", type=float, default=1e-5)
    arg_parser.add_argument("--model_rope_theta", type=int_or_none, default=10000)
    arg_parser.add_argument("--model_tie_embeddings", action="store_true",
                            help="share one matrix between token embeddings and lm_head")
    arg_parser.add_argument("--model_qk_norm", action="store_true", help="RMSNorm on each head's queries and keys, before RoPE")
    arg_parser.add_argument("--model_logit_soft_cap", type=float, default=None, help="squash output logits to c * tanh(logits / c); unset turns it off")
    arg_parser.add_argument("--model_compile", action="store_true")
    arg_parser.add_argument("--gradient_max_l2_norm", type=float, default=1.0)
    arg_parser.add_argument("--gradient_eps", type=float, default=1e-6)
    arg_parser.add_argument("--n", type=int, default=10)
    arg_parser.add_argument("--w", type=int, default=5, help="warmup iters before measurement")
    arg_parser.add_argument("--device", type=str, default="mps")
    arg_parser.add_argument("--batch_size", type=int, default=4)
    arg_parser.add_argument("--bf16", action="store_true")
    args = arg_parser.parse_args()

    if args.optimizer_warmup_iters is None:
        args.optimizer_warmup_iters = args.w
    if args.optimizer_cosine_cycle_iters is None:
        args.optimizer_cosine_cycle_iters = args.n

    assert args.optimizer_lr_min <= args.optimizer_lr_max
    assert args.n >= 2

    benchmark(args)
