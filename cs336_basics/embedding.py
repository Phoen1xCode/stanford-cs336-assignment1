import torch 
import torch.nn as nn

from typing import Optional

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] | None = None,
        dtype: Optional[torch.dtype] | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=1.0, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]