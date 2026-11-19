from typing import Any

import torch.distributed as dist
import torch.nn

from cs336_basics.model import Embedding, Linear


class FullyShardedDataParallel(torch.nn.Module):
    module: torch.nn.Module
    layers: dict[int, torch.nn.Module]
    pending_gradients: list[tuple[torch.Tensor, torch.Tensor, Any]]  # tuple[param, out_tensor, handle]
    pending_unsharded_gradients: list[Any]  # handle
    compute_dtype: torch.dtype | None

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.layers = {}
        self.pending_gradients = []
        self.pending_unsharded_gradients = []
        self.compute_dtype = compute_dtype
        self._shard_weights()
        self._register_hooks(compute_dtype)

    def forward(self, *inputs, **kwargs):
        for i in range(2):
            if i in self.layers:
                layer = self.layers[i]
                handle, out_tensor, in_tensor = self._request_layer_gather(layer, self.compute_dtype)
                layer._fwd_gather_handle = handle
                layer._fwd_gather_out = out_tensor
                layer._fwd_gather_in = in_tensor

        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for param, out, handle in self.pending_gradients:
            handle.wait()
            param.grad.data = out.to(torch.float32)
        self.pending_gradients.clear()
        for handle in self.pending_unsharded_gradients:
            handle.wait()
        self.pending_unsharded_gradients.clear()

    def _request_layer_gather(self, layer: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        in_tensor: torch.Tensor = layer.weight.data.to(compute_dtype)
        shard_size: int = in_tensor.shape[0]
        world_size: int = dist.get_world_size()
        full_size: int = shard_size * world_size
        out_tensor: torch.Tensor = torch.empty((full_size, *in_tensor.shape[1:]), dtype=in_tensor.dtype, device=in_tensor.device)
        handle = dist.all_gather_into_tensor(out_tensor, in_tensor, async_op=True)
        return handle, out_tensor, in_tensor

    def _register_hooks(self, compute_dtype: torch.dtype | None = None):
        for mod in self.module.modules():
            if not isinstance(mod, (Linear, Embedding)):
                continue

            def make_fwd_pre(dt):
                def hook(m, inp):
                    m._saved_fp32 = m.weight.data
                    if not hasattr(m, "_layer_pos"):
                        m._layer_pos = len(self.layers)
                        self.layers[m._layer_pos] = m
                    if hasattr(m, "_fwd_gather_handle"):
                        m._fwd_gather_handle.wait()
                        del m._fwd_gather_handle
                        del m._fwd_gather_in
                        m.weight.data = m._fwd_gather_out
                        del m._fwd_gather_out
                    else:
                        handle, out_tensor, _ = self._request_layer_gather(m, dt)
                        handle.wait()
                        m.weight.data = out_tensor

                return hook

            def make_fwd_post(dt):
                def hook(m, inp, out):
                    m.weight.data = m._saved_fp32
                    del m._saved_fp32
                    if hasattr(m, "_layer_pos") and m._layer_pos + 2 in self.layers:
                        future_layer = self.layers[m._layer_pos + 2]
                        handle, out_tensor, in_tensor = self._request_layer_gather(future_layer, dt)
                        future_layer._fwd_gather_handle = handle
                        future_layer._fwd_gather_out = out_tensor
                        future_layer._fwd_gather_in = in_tensor

                return hook

            mod.register_forward_pre_hook(make_fwd_pre(compute_dtype))
            mod.register_forward_hook(make_fwd_post(compute_dtype))

            if isinstance(mod, Linear):
                def make_bwd_pre(dt):
                    def hook(m, grad_out):
                        m._saved_fp32_bwd = m.weight.data
                        handle, out_tensor, _ = self._request_layer_gather(m, dt)
                        handle.wait()
                        m.weight.data = out_tensor

                    return hook

                mod.register_full_backward_pre_hook(make_bwd_pre(compute_dtype))

            def make_grad_hook(m, is_linear):
                def hook(param: torch.Tensor) -> None:
                    if is_linear and hasattr(m, "_saved_fp32_bwd"):
                        m.weight.data = m._saved_fp32_bwd
                        del m._saved_fp32_bwd
                    if param.grad is not None:
                        world_size: int = dist.get_world_size()
                        dim: int = param.grad.data.shape[0]
                        assert dim % world_size == 0
                        shard_size: int = dim // world_size
                        out: torch.Tensor = torch.empty((shard_size, *param.grad.data.shape[1:]), dtype=param.grad.data.dtype, device=param.grad.data.device)
                        handle = dist.reduce_scatter_tensor(out, param.grad.data, op=dist.ReduceOp.AVG, async_op=True)
                        self.pending_gradients.append((param, out, handle))

                return hook

            mod.weight.register_post_accumulate_grad_hook(make_grad_hook(mod, isinstance(mod, Linear)))

        for param in self.module.parameters():
            sharded: bool = getattr(param, "sharded", False)
            if sharded or not param.requires_grad:
                continue

            def unsharded_grad_hook(param: torch.Tensor):
                if param.grad is not None:
                    handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                    self.pending_unsharded_gradients.append(handle)

            param.register_post_accumulate_grad_hook(unsharded_grad_hook)

    def _shard_weights(self):
        for mod in self.module.modules():
            if not isinstance(mod, (Linear, Embedding)):
                continue
            rank: int = dist.get_rank()
            world_size: int = dist.get_world_size()
            dim: int = mod.weight.data.shape[0]
            assert dim % world_size == 0
            shard_size: int = dim // world_size
            mod.weight.data = mod.weight.data[rank * shard_size: (rank + 1) * shard_size].clone()
            mod.weight.sharded = True

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for name, param in self.module.named_parameters():
            sharded: bool = getattr(param, "sharded", False)
            if not sharded:
                output[name] = param
            else:
                shard: torch.Tensor = param.data
                shard_size: int = shard.shape[0]
                world_size: int = dist.get_world_size()
                full_size: int = shard_size * world_size
                out: torch.Tensor = torch.empty((full_size, *shard.shape[1:]), dtype=shard.dtype, device=shard.device)
                dist.all_gather_into_tensor(out, shard)
                output[name] = out

        return output
