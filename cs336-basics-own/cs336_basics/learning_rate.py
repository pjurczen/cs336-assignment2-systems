import math


def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if t < warmup_iters:
        return t / warmup_iters * lr_max
    if warmup_iters <= t <= cosine_cycle_iters:
        return lr_min + 1 / 2 * (1 + math.cos((t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (lr_max - lr_min)
    return lr_min


def cosine_lr_schedule_by_time(t: int, elapsed_seconds: float, budget_seconds: float, lr_max: float, lr_min: float, warmup_iters: int) -> float:
    # Like cosine_lr_schedule, but the cosine follows the share of the time budget used instead of the step count.
    # Warmup stays in steps; it's short, so the cosine starting at time 0 instead of after warmup barely matters.
    if t < warmup_iters:
        return t / warmup_iters * lr_max
    if elapsed_seconds <= budget_seconds:
        return lr_min + 1 / 2 * (1 + math.cos(elapsed_seconds / budget_seconds * math.pi)) * (lr_max - lr_min)
    return lr_min
