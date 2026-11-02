import torch
from einops import rearrange


# inputs (batch_size, vocab_size)
# targets (batch_size)
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    targets_per_batch = rearrange(targets, "... -> ... 1")
    inputs_norm = inputs.subtract(inputs_max)
    target_logit = torch.gather(inputs_norm, -1, targets_per_batch)
    result = -1 * (target_logit - inputs_norm.exp().sum(dim=-1, keepdim=True).log())
    return result.mean()


def cross_entropy_z_loss(inputs: torch.Tensor, targets: torch.Tensor, lambda_w: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    targets_per_batch = rearrange(targets, "... -> ... 1")
    inputs_norm = inputs.subtract(inputs_max)
    target_logit = torch.gather(inputs_norm, -1, targets_per_batch)
    logitsum = inputs_norm.exp().sum(dim=-1, keepdim=True).log()
    z = inputs_max + logitsum
    cross_entropy_loss = -1 * (target_logit - logitsum)
    z_loss = lambda_w * z ** 2
    return cross_entropy_loss.mean(), z_loss.mean()
