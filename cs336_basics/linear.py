import torch
import torch.nn as nn
from typing import Optional

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        
        self.W = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization
        std = 2 / (self.in_features + self.out_features) ** 0.5
        torch.nn.init.trunc_normal_(self.W, std=std, a= -3 * std, b= 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.t()
        