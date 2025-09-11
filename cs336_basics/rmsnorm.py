import torch
import torch.nn as nn

from typing import Optional


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] | None = None,
        dtype: Optional[torch.device] | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.W = nn.Parameter(torch.ones((d_model), **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # x.shape = (batch_size, sequence_length, d_model)
        # mean_square.shape = (batch_size, sequence_length, 1)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True) 

        # RMS: x / sqrt(mean_square + eps)
        # torch.rsqrt: 1 / sqrt(x)
        x = x * torch.rsqrt(mean_square + self.eps)

        x = self.W * x

        return x.to(input_dtype)
