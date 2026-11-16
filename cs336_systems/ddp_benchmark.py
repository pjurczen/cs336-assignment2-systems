import argparse
import contextlib
import os
import statistics
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.adamw_optimizer import AdamWOptimizer
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.muon_optimizer import MuonOptimizer
from cs336_basics.transformer_lm import TransformerLM
from distributed_data_parallel import DistributedDataParallel


def _setup_process_group(rank: int, world_size: int, device: str):
    backend: str = "nccl" if device.startswith("cuda") else "gloo"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def benchmark(rank: int, world_size: int, args: argparse.Namespace):
    device = _setup_process_group(rank=rank, world_size=world_size, device=args.device)
    dist.barrier()

    raw_model: TransformerLM = TransformerLM(
        vocab_size=args.model_vocab_size,
        context_length=args.model_context_length,
        num_layers=args.model_num_layers,
        d_model=args.model_d_model,
        num_heads=args.model_num_heads,
        d_ff=args.model_d_ff,
        eps=args.model_eps,
        theta=args.model_rope_theta,
        device=torch.device(device),
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
    ddp_model: DistributedDataParallel = DistributedDataParallel(model)

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

    inputs, targets = get_random_input_batch(batch_size=args.batch_size // world_size, context_length=args.model_context_length, vocab_size=args.model_vocab_size, device=args.device, dtype=torch.long)

    forward_timing = []
    backward_timing = []
    optimizers_timing = []
    communication_timing = []
    total_timing = []

    for i in range(1, args.w + 1):
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.bf16):
            with torch.no_grad() if args.forward_only else contextlib.nullcontext():
                logits = ddp_model(inputs)
            if not args.forward_only:
                loss: torch.Tensor = cross_entropy(logits, targets)

        if not args.forward_only:
            loss.backward()

            ddp_model.finish_gradient_synchronization()

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

        if i == args.w and rank == 0:
            print("Warmup done")

    synchronize(args.device)
    dist.barrier()

    for i in range(1, args.n + 1):

        t0 = timeit.default_timer()
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.bf16):
            with torch.no_grad() if args.forward_only else contextlib.nullcontext():
                logits = ddp_model(inputs)
        synchronize(args.device)

        t1 = timeit.default_timer()
        if not args.forward_only:
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.bf16):
                loss: torch.Tensor = cross_entropy(logits, targets)
            synchronize(args.device)

            t2 = timeit.default_timer()
            loss.backward()
            synchronize(args.device)

            t3 = timeit.default_timer()
            ddp_model.finish_gradient_synchronization()
            synchronize(args.device)

            t4 = timeit.default_timer()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            synchronize(args.device)

            t5 = timeit.default_timer()

            backward_timing.append(t3 - t2)
            optimizers_timing.append(t5 - t4)
            communication_timing.append(t4 - t3)
            total_timing.append(t5 - t0)

        forward_timing.append(t1 - t0)

    all_forward_timing = [None] * world_size
    all_backward_timing = [None] * world_size
    all_optimizers_timing = [None] * world_size
    all_communication_timing = [None] * world_size
    all_total_timing = [None] * world_size

    dist.all_gather_object(all_forward_timing, forward_timing)
    dist.all_gather_object(all_backward_timing, backward_timing)
    dist.all_gather_object(all_optimizers_timing, optimizers_timing)
    dist.all_gather_object(all_communication_timing, communication_timing)
    dist.all_gather_object(all_total_timing, total_timing)

    forward_timing = [t for ts in all_forward_timing for t in ts]
    backward_timing = [t for ts in all_backward_timing for t in ts]
    optimizers_timing = [t for ts in all_optimizers_timing for t in ts]
    communication_timing = [t for ts in all_communication_timing for t in ts]
    total_timing = [t for ts in all_total_timing for t in ts]

    if rank == 0:
        print(f"Warmup iters: {args.w}")
        print(f"Measurement iters: {args.n}")

        print(f"Forward pass: {statistics.mean(forward_timing) * 1000:.2f} ± {statistics.stdev(forward_timing) * 1000:.2f} ms")
        print(f"Forward pass values: {forward_timing}")
        if not args.forward_only:
            print(f"Backward pass: {statistics.mean(backward_timing) * 1000:.2f} ± {statistics.stdev(backward_timing) * 1000:.2f} ms")
            print(f"Backward pass values: {backward_timing}")

            print(f"Optimizer step: {statistics.mean(optimizers_timing) * 1000:.2f} ± {statistics.stdev(optimizers_timing) * 1000:.2f} ms")
            print(f"Optimizer step values: {optimizers_timing}")

            print(f"Communication: {statistics.mean(communication_timing) * 1000:.2f} ± {statistics.stdev(communication_timing) * 1000:.2f} ms")
            print(f"Communication values: {communication_timing}")

            print(f"Total: {statistics.mean(total_timing) * 1000:.2f} ± {statistics.stdev(total_timing) * 1000:.2f} ms")
            print(f"Total values: {total_timing}")

    dist.destroy_process_group()


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
    arg_parser.add_argument("--n", type=int, default=10)
    arg_parser.add_argument("--w", type=int, default=5, help="warmup iters before measurement")
    arg_parser.add_argument("--device", type=str, default="cpu")
    arg_parser.add_argument("--batch_size", type=int, default=4, help="Batch size across all workers together")
    arg_parser.add_argument("--bf16", action="store_true")
    arg_parser.add_argument("--forward_only", action="store_true")
    args = arg_parser.parse_args()

    assert args.n >= 2

    world_size = 2
    assert args.batch_size % world_size == 0

    mp.spawn(
        benchmark,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
