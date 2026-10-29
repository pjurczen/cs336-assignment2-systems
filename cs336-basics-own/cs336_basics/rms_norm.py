import torch


class RMSNorm(torch.nn.Module):
    weight: torch.nn.Parameter  # (d_model, )
    eps: float

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    # x (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_sqrd_mean = x.pow(2).mean(dim=-1, keepdim=True)  # (batch_size, sequence_length, 1)
        rms = torch.sqrt(x_sqrd_mean + self.eps)  # (batch_size, sequence_length, 1)
        # (batch_size, sequence_length, d_model) * (d_model, ) / (batch_size, sequence_length, 1)
        result = x * self.weight / rms  # (batch_size, sequence_length, d_model)
        return result.to(in_dtype)
