import torch
import torch.nn as nn

from typing import Optional
from .linear import Linear

def SiLU(x: torch.Tensor) -> torch.Tensor:
    in_type = x.dtype
    x = x.to(torch.float32)
    return (x * torch.sigmoid(x)).to(in_type)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] | None = None,
        dtype: Optional[torch.dtype] | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = Linear(d_model, d_ff, **factory_kwargs)
        self.W2 = Linear(d_ff, d_model, **factory_kwargs)
        self.W3 = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = SiLU(self.W1(x)) * self.W3(x)
        return self.W2(gate)


