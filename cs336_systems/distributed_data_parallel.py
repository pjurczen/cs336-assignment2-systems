from typing import Any

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


class DistributedDataParallelBatch(torch.nn.Module):
    module: torch.nn.Module

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        for p in module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self):
        tensors = [p.grad for p in self.module.parameters() if p.grad is not None]
        tensors_flat = torch._utils._flatten_dense_tensors(tensors)
        dist.all_reduce(tensors_flat, op=dist.ReduceOp.AVG)
        reduced = torch._utils._unflatten_dense_tensors(flat=tensors_flat, tensors=tensors)
        for t, r in zip(tensors, reduced):
            t.data = r.data


class DistributedDataParallelOverlapParams(torch.nn.Module):
    module: torch.nn.Module
    handles: list[Any]

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        def communication_hook(param: torch.Tensor) -> None:
            h = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.handles.append(h)

        for p in module.parameters():
            dist.broadcast(p.data, src=0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(communication_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
