import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ActiveKwargs(nn.Module):
    """Adapts a self-attention or ffn module to be a part of torch.nn.Sequential"""

    def __init__(self, module: nn.Module, active_kwargs=()):
        super().__init__()
        self.module, self.active_kwargs = module, active_kwargs

    def forward(self, input, **kwargs):
        active_kwargs = {key: value for key, value in kwargs.items() if key in self.active_kwargs}
        return self.module(input, **active_kwargs)


class SequentialWithKwargs(nn.Sequential):
    def __init__(self, *modules: ActiveKwargs):
        for module in modules:
            assert isinstance(module, ActiveKwargs)
        super().__init__(*modules)
        self.gradient_checkpointing = False

    def forward(self, input: torch.Tensor, **kwargs):
        for module in self:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                input, *_extras = checkpoint(module, input, **kwargs)
            else:
                input, *_extras = module(input, **kwargs)
        return input
