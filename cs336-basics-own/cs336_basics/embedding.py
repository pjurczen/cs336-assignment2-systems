import torch


class EmbeddingLookup(torch.nn.Module):
    weight: torch.nn.Parameter
    scale_factor: float

    def __init__(self, weight: torch.nn.Parameter, scale_factor: float = 1.0):
        super().__init__()
        self.weight = weight
        self.scale_factor = scale_factor

    # token_ids: torch.LongTensor (batch_size, sequence_length)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids] * self.scale_factor


class Embedding(EmbeddingLookup):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        weight = torch.nn.parameter.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(tensor=weight, mean=0, std=1, a=-3, b=3)
        super().__init__(weight)
