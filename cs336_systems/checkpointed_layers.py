import torch
from torch.utils.checkpoint import checkpoint


class CheckpointedLayers(torch.nn.Module):
    layers: torch.nn.ModuleList

    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._run, x, use_reentrant=False)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
