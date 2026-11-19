import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.fsdp import FullyShardedDataParallel
from tests.test_fsdp import ToyFSDPModel


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29520"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)
    x = torch.randint(0, 100, (4, 16))
    model = ToyFSDPModel(vocab_size=100, d_model=64, d_ff=128)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    fsdp = FullyShardedDataParallel(deepcopy(model))
    full = fsdp.gather_full_params()
    y = model(x)
    y_fdsp = fsdp(x)
    if rank == 0:
        for name, p in fsdp.module.named_parameters():
            print(f"{name}: held {tuple(p.shape)}, gathered {tuple(full[name].shape)}, equals original: {torch.equal(full[name], original[name])}", flush=True)
        print(f"Forward pass with fsdp equals unwrapped: {torch.equal(y, y_fdsp)}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(fn=worker, args=(2,), nprocs=2, join=True)
