from collections import defaultdict
from typing import Type, Any, Iterable, Callable

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    optimizer: torch.optim.Optimizer | None
    optimizer_cls: Type[torch.optim.Optimizer]
    params: dict[int, list[torch.nn.parameter.Parameter]]
    defaults: dict[str, Any]
    cur_rank: int

    def __init__(self, params: Iterable[torch.nn.parameter.Parameter], optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.cur_rank = 0
        self.defaults = kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer = None
        self.params = defaultdict(list)
        super().__init__(params, kwargs)

    def step(self, closure: Callable | None = None, **kwargs):
        if self.optimizer:
            self.optimizer.step(closure, **kwargs)
        for rank, params in self.params.items():
            for p in params:
                dist.broadcast(p.data, src=rank)

    # This is called during construction of the sharded optimizer by
    # the super-class constructor and may also be called during training (e.g., for gradually
    # unfreezing layers in a model)
    def add_param_group(self, param_group: dict[str, Any]):
        self.param_groups.append(param_group)
        rank: int = dist.get_rank()
        group = []
        for p in param_group["params"]:
            self.params[self.cur_rank].append(p)
            if rank == self.cur_rank:
                group.append(p)
            self.cur_rank += 1
            self.cur_rank = self.cur_rank % dist.get_world_size()
        group_to_add = {**param_group, "params": group}
        if group:
            if self.optimizer:
                self.optimizer.add_param_group(group_to_add)
            else:
                self.optimizer = self.optimizer_cls([group_to_add], **self.defaults)
