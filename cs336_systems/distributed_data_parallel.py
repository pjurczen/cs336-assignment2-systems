import torch
import torch.distributed as dist


class DistributedDataParallel(torch.nn.Module):
    module: torch.nn.Module

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        for p in module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self):
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
