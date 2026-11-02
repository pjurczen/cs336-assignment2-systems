import argparse
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import numpy.typing as npt
import torch.optim
import wandb

from cs336_basics.adamw_optimizer import AdamWOptimizer
from cs336_basics.bpe_tokenizer import read_tokenizer_config
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy, cross_entropy_z_loss
from cs336_basics.data_loading import get_batch, get_random_indices, get_nonoverlapping_indices
from cs336_basics.gradient import gradient_clipping
from cs336_basics.learning_rate import cosine_lr_schedule, cosine_lr_schedule_by_time
from cs336_basics.muon_optimizer import MuonOptimizer
from cs336_basics.transformer_lm import TransformerLM


def training(args: argparse.Namespace):
    iteration: int = 1

    run_dir = os.path.join(args.runs_dir, args.run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run directory: {run_dir}")

    # Live dashboard, off unless --wandb_project is given. Needs WANDB_API_KEY or a prior `wandb login`.
    wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args)) if args.wandb_project else None

    model: torch.nn.Module = TransformerLM(
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
    compiled_model: torch.nn.Module = torch.compile(model, backend=compile_backend)

    # With --optimizer_muon, the 2D matrices inside the Transformer blocks go to Muon; everything else stays on AdamW.
    matrix_params = []
    other_params = []
    for name, p in model.named_parameters():
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

    elapsed_time = None
    if args.checkpoint_src:
        print("Loading checkpoint...")
        iteration, elapsed_time = load_checkpoint(args.checkpoint_src, model, adamw, muon)

        print(f"Resuming from iteration {iteration}/{args.iter}")
        iteration += 1

    train_dataset = np.load(file=args.train_data, mmap_mode='r')
    val_dataset = np.load(file=args.val_data, mmap_mode='r') if args.val_data else None

    iter_val_indices: npt.NDArray[int] = get_nonoverlapping_indices(val_dataset, args.model_context_length,
                                                                    args.val_batches * args.batch_size) if args.val_data else None
    final_val_indices: npt.NDArray[int] = get_nonoverlapping_indices(val_dataset, args.model_context_length) if args.val_data else None

    with open(os.path.join(run_dir, "metrics.jsonl"), "a") as metric_file:

        t0 = time.perf_counter()
        if elapsed_time is not None:
            t0 -= elapsed_time

        # With a time budget, training ends when it's used up (minus the reserve for the final validation and checkpoint),
        # and the cosine follows the elapsed time instead of the step count. --iter is then only an upper limit.
        budget_seconds = (args.time_budget_minutes - args.time_reserve_minutes) * 60 if args.time_budget_minutes is not None else None

        for it in range(iteration, args.iter + 1):
            elapsed = time.perf_counter() - t0  # no synchronize: the CPU runs only a few steps ahead of the GPU
            last_step = it == args.iter or (budget_seconds is not None and elapsed >= budget_seconds)

            indices: npt.NDArray[int] = get_random_indices(dataset=train_dataset, batch_size=args.batch_size, context_length=args.model_context_length)
            inputs, targets = get_batch(dataset=train_dataset, indices=indices, context_length=args.model_context_length, device=args.device, dtype=torch.long)
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.bf16):
                prediction: torch.Tensor = compiled_model(inputs)

                # No float32 copy of the logits here: it costs batch × context × vocab × 4 bytes (3.9 GiB at 64 × 512 × 32K).
                # Autocast still runs exp, log and sum in float32. evaluate() keeps the copy, since it holds no activations.
                cross_entropy_loss, z_loss = cross_entropy_z_loss(prediction, targets, lambda_w=args.z_loss_weight)
                loss = cross_entropy_loss + z_loss
            loss.backward()

            l2_norm: torch.Tensor = gradient_clipping(parameters=model.parameters(), max_l2_norm=args.gradient_max_l2_norm, eps=args.gradient_eps)

            if (it <= args.optimizer_warmup_iters and it % 10 == 0) or it % args.log_interval == 0 or last_step:
                print(f"{it}/{args.iter} Training loss: {cross_entropy_loss.item()}")
            if it % (10 * args.log_interval) == 0 or last_step:
                print(f"{it}/{args.iter} L2-norm: {l2_norm.item()}")

            if budget_seconds is not None:
                lr: float = cosine_lr_schedule_by_time(
                    t=it,
                    elapsed_seconds=elapsed,
                    budget_seconds=budget_seconds,
                    lr_max=args.optimizer_lr_max,
                    lr_min=args.optimizer_lr_min,
                    warmup_iters=args.optimizer_warmup_iters,
                )
            else:
                lr: float = cosine_lr_schedule(
                    t=it,
                    lr_max=args.optimizer_lr_max,
                    lr_min=args.optimizer_lr_min,
                    warmup_iters=args.optimizer_warmup_iters,
                    cosine_cycle_iters=args.optimizer_cosine_cycle_iters,
                )
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()

            if (it <= args.optimizer_warmup_iters and it % 10 == 0) or it % args.log_interval == 0 or last_step:
                synchronize(args.device)
                t1 = time.perf_counter()

                print(f"Elapsed time: {(t1 - t0):.0f} s")

                metrics: dict = {
                    "step": it, "time": round(t1 - t0, 2), "train_loss": cross_entropy_loss.item(), "lr": lr, "grad_norm": l2_norm.item(),
                    "peak_memory_gb": peak_memory_gb(args.device),
                }
                metric_file.write(json.dumps(metrics) + "\n")
                metric_file.flush()
                log_to_wandb(wandb_run, metrics)

            if val_dataset is not None and (it % args.val_interval == 0 or last_step):
                if last_step:
                    val_loss: float = evaluate(compiled_model, val_dataset, final_val_indices, args.batch_size, args.model_context_length, args.device,
                                               bf16=args.bf16, full_dataset_eval=True)
                elif it % args.val_interval == 0:
                    val_loss: float = evaluate(compiled_model, val_dataset, iter_val_indices, args.batch_size, args.model_context_length, args.device,
                                               bf16=args.bf16)
                t2 = time.perf_counter()

                metrics: dict = {
                    "step": it, "time": round(t2 - t0, 2), "val_loss": val_loss,
                }
                metric_file.write(json.dumps(metrics) + "\n")
                metric_file.flush()
                log_to_wandb(wandb_run, metrics)

                print(f"{it}/{args.iter} Validation loss: {val_loss}")

            if it % args.checkpoints_iter == 0 or last_step:
                t3 = time.perf_counter()
                save_checkpoint(model, adamw, it, os.path.join(checkpoints_dir, f"ckpt_{it}.pt"), t3 - t0, muon)
                print(f"{it}/{args.iter} Saving checkpoint...")

            if last_step:
                break

    if wandb_run is not None:
        wandb_run.finish()


def evaluate(model: torch.nn.Module, dataset: npt.NDArray, indices: npt.NDArray, batch_size: int, context_length: int, device: str,
             bf16: bool = False, full_dataset_eval: bool = False) -> float:
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
            val_loss_sum: float = 0
            token_count: int = 0
            for idx in range(0, len(indices), batch_size):
                batch = indices[idx: (idx + batch_size)]
                val_inputs, val_targets = get_batch(dataset=dataset, indices=batch, context_length=context_length, device=device, dtype=torch.long)

                val_prediction = model(val_inputs)
                val_loss_sum += cross_entropy(val_prediction.float(), val_targets).item() * (len(batch) * context_length)
                token_count += len(batch) * context_length

            if full_dataset_eval:
                tail_length: int = (len(dataset) - 1) % context_length
                if tail_length > 0:
                    start = len(dataset) - 1 - tail_length
                    val_inputs, val_targets = get_batch(dataset=dataset, indices=np.array([start]), context_length=tail_length, device=device, dtype=torch.long)
                    val_prediction = model(val_inputs)
                    val_loss_sum += cross_entropy(val_prediction.float(), val_targets).item() * tail_length
                    token_count += tail_length

            val_loss = val_loss_sum / token_count

            return val_loss


def synchronize(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def peak_memory_gb(device: str) -> float | None:
    # Most GPU memory held by tensors so far in this run (CUDA only).
    if device.startswith("cuda"):
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return None


def log_to_wandb(wandb_run, metrics: dict):
    # Same metrics as metrics.jsonl; wandb takes the step separately and skips missing values.
    if wandb_run is not None:
        wandb_run.log({key: value for key, value in metrics.items() if key != "step" and value is not None}, step=metrics["step"])


def git_commit() -> str | None:
    # Inside the training image there is no Git; scripts/build_image.sh bakes the version in as $GIT_COMMIT.
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"]).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


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
    arg_parser.add_argument("--optimizer_muon", action="store_true", help="Muon for the 2D matrices inside the blocks, AdamW for the rest")
    arg_parser.add_argument("--model_context_length", type=int, default=1024)
    arg_parser.add_argument("--model_num_layers", type=int, default=48)
    arg_parser.add_argument("--model_d_model", type=int, default=1600)
    arg_parser.add_argument("--model_d_ff", type=int, default=4288)
    arg_parser.add_argument("--model_num_heads", type=int, default=25)
    arg_parser.add_argument("--model_eps", type=float, default=1e-5)
    arg_parser.add_argument("--model_rope_theta", type=int_or_none, default=10000)
    arg_parser.add_argument("--model_tie_embeddings", action="store_true", help="share one matrix between token embeddings and lm_head")
    arg_parser.add_argument("--model_qk_norm", action="store_true", help="RMSNorm on each head's queries and keys, before RoPE")
    arg_parser.add_argument("--model_logit_soft_cap", type=float, default=None, help="squash output logits to c * tanh(logits / c); unset turns it off")
    arg_parser.add_argument("--gradient_max_l2_norm", type=float, default=1.0)
    arg_parser.add_argument("--gradient_eps", type=float, default=1e-6)
    arg_parser.add_argument("--z_loss_weight", type=float, default=0.0, help="weight of the z-loss added to the training loss; 0 turns it off")
    arg_parser.add_argument("--iter", type=int, default=100)
    arg_parser.add_argument("--device", type=str, default="mps")
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--tokenizer_dir", type=str, required=True)
    arg_parser.add_argument("--train_data", type=str, required=True)
    arg_parser.add_argument("--val_data", type=str)
    arg_parser.add_argument("--val_batches", type=int, default=32)
    arg_parser.add_argument("--val_interval", type=int, default=200)
    arg_parser.add_argument("--log_interval", type=int, default=100)
    arg_parser.add_argument("--runs_dir", type=str, default="runs")
    arg_parser.add_argument("--run_name", type=str, default=None)
    arg_parser.add_argument("--checkpoints_iter", type=int, default=100)
    arg_parser.add_argument("--checkpoint_src", type=str, default=None)
    arg_parser.add_argument("--wandb_project", type=str, default=None)
    arg_parser.add_argument("--bf16", action="store_true")
    arg_parser.add_argument("--time_budget_minutes", type=float, default=None,
                            help="train for this long instead of --iter steps (--iter becomes an upper limit); needs --optimizer_warmup_iters")
    arg_parser.add_argument("--time_reserve_minutes", type=float, default=0.0,
                            help="part of the time budget kept free after training, for the final validation pass and checkpoint")
    args = arg_parser.parse_args()
    args.git_commit = git_commit()  # recorded in config.json and the wandb config

    if args.time_budget_minutes is not None and args.optimizer_warmup_iters is None:
        arg_parser.error("--time_budget_minutes needs an explicit --optimizer_warmup_iters (the step count isn't known in advance)")
    if args.optimizer_warmup_iters is None:
        args.optimizer_warmup_iters = args.iter // 50
    if args.optimizer_cosine_cycle_iters is None:
        args.optimizer_cosine_cycle_iters = args.iter
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args.model_vocab_size = read_tokenizer_config(args.tokenizer_dir)["vocab_size"]

    assert args.optimizer_lr_min <= args.optimizer_lr_max
    assert args.optimizer_warmup_iters < args.optimizer_cosine_cycle_iters <= args.iter

    training(args)
