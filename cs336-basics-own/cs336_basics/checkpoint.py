import os
from typing import BinaryIO, IO, Optional

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
        elapsed_time: float | None = None,
        muon_optimizer: torch.optim.Optimizer | None = None,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if elapsed_time is not None:
        checkpoint["elapsed_time"] = elapsed_time
    if muon_optimizer is not None:
        checkpoint["muon_optimizer"] = muon_optimizer.state_dict()
    torch.save(checkpoint, out)


def upgrade_state_dict(state_dict: dict[str, torch.Tensor]):
    """Converts model weights saved by older code to the current layout, in place."""
    # Checkpoints saved from the torch.compile wrapper prefix every key.
    consume_prefix_in_state_dict_if_present(state_dict, "_orig_mod.")


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        muon_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, Optional[float]]:
    # Load onto the model's device, so a checkpoint saved on CUDA can be resumed on MPS and vice versa.
    checkpoint: dict[str, any] = torch.load(src, map_location=next(model.parameters()).device)
    upgrade_state_dict(checkpoint["model"])
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if muon_optimizer is not None:
        if "muon_optimizer" not in checkpoint:
            raise ValueError("Resuming with Muon, but the checkpoint was trained without it")
        muon_optimizer.load_state_dict(checkpoint["muon_optimizer"])
    return checkpoint["iteration"], checkpoint.get("elapsed_time", None)
