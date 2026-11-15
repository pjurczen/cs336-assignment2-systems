import os
import statistics
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

data_sizes = [262144, 2621440, 26214400, 268435456]  # 1MB, 10MB, 100MB, 1GB
gpu_counts = [2, 4, 6]
device = "cuda"
WARMUP_ITERS = 5
TIMING_ITERS = 10


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if device == "cuda":
        torch.cuda.set_device(rank)
    if device == "cuda":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_benchmark(rank, world_size: int, input_size: int):
    setup(rank, world_size)
    all_timings = [None] * world_size
    local_timings = []
    for _ in range(WARMUP_ITERS):
        data = torch.randn((input_size,), dtype=torch.float32, device=device)
        dist.all_reduce(data, async_op=False)

    for _ in range(TIMING_ITERS):
        data = torch.randn((input_size,), dtype=torch.float32, device=device)
        synchronize(device)

        t0 = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        synchronize(device)
        t1 = timeit.default_timer()

        delta = t1 - t0
        local_timings.append(delta)

    dist.all_gather_object(all_timings, local_timings)
    timings = [t for ts in all_timings for t in ts]
    if rank == 0:
        print(f"World size: {world_size}, input size: {input_size * 4 / 1024 ** 2} MiB, timing: {statistics.mean(timings) * 1000:.2f} ± {statistics.stdev(timings) * 1000:.2f} ms")

    dist.destroy_process_group()


def synchronize(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda" or device == "cuda:0":
        torch.cuda.synchronize()


if __name__ == "__main__":
    for world_size in gpu_counts:
        for input_size in data_sizes:
            mp.spawn(fn=distributed_benchmark, args=(world_size, input_size,), nprocs=world_size, join=True)
