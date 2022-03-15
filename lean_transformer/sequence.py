"""
A module that implements sequential model type with optional keyword arguments.
When using gradient checkpoints or reversible sequential, keyword arguments should NOT require grad.
"""
from typing import Callable, Sequence

import torch
from lean_transformer.utils import get_logger
from revlib import ReversibleModule, ReversibleSequential
from torch import nn as nn
from torch.utils.checkpoint import checkpoint

logger = get_logger(__name__)


class ActiveKwargs(nn.Module):
    """
    A module with selective kwargs, compatible with sequential, gradient checkpoints and
    Usage: ony use this as a part of SequentialWithKwargs or ReversibleWithKwargs
    """

    def __init__(self, module: nn.Module, active_keys: Sequence[str], use_first_output: bool = False):
        super().__init__()
        self.module, self.active_keys, self.use_first_output = module, set(active_keys), use_first_output

    def forward(self, input: torch.Tensor, *args, **kwargs):
        kwargs = {key: value for key, value in kwargs.items() if key in self.active_keys}
        output = self.module(input, *args, **kwargs)
        if self.use_first_output and not isinstance(output, torch.Tensor):
            output = output[0]
        return output


class SequentialWithKwargs(nn.Sequential):
    def __init__(self, *modules: ActiveKwargs):
        for module in modules:
            assert isinstance(module, ActiveKwargs) or (
                isinstance(module, ReversibleModule) and any(isinstance(m, ActiveKwargs) for m in module.modules())
            )
        super().__init__(*modules)
        self.gradient_checkpointing = False

    def forward(self, input: torch.Tensor, *args, **kwargs):
        kwarg_keys, kwarg_values = zip(*kwargs.items()) if (self.gradient_checkpointing and kwargs) else ([], [])
        for module in self:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                # pack kwargs with args since gradient checkpoint does not support kwargs
                input = checkpoint(self._checkpoint_forward, module, input, kwarg_keys, *kwarg_values, *args)
            else:
                input = module(input, *args, **kwargs)
        return input

    def _checkpoint_forward(self, module: Callable, input: torch.Tensor, kwarg_keys: Sequence[str], *etc):
        kwargs = {key: etc[i] for i, key in enumerate(kwarg_keys)}
        args = etc[len(kwarg_keys) :]
        return module(input, *args, **kwargs)


class ReversibleWithKwargs(ReversibleSequential):
    def __init__(self, *modules, **kwargs):
        for module in modules:
            assert isinstance(module, ActiveKwargs) or (
                isinstance(module, ReversibleModule) and any(isinstance(m, ActiveKwargs) for m in module.modules())
            )
        super().__init__(*modules, **kwargs)
        wrapped_modules = getattr(self, 'stem', [m for m in self.children() if isinstance(m, ReversibleModule)])
        assert len(wrapped_modules) == len(modules)
        self.stem = SequentialWithKwargs(*wrapped_modules)

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        inp1 = inp0 = input.to(torch.float32)  # enforce upcasting residuals to fp32
        zeros = torch.zeros_like(inp0)
        out0, out1 = self.replace_grad(*self.stem((inp0, inp1, zeros, zeros), *args, **kwargs))
        # note: we initialize both sides of reversible sequence with the same input (e.g. embeddings)
        # and combine them to get something resembling a normal residual sum. More specifically,
        # out1 = input + f1 + f2 + ... + fn  -- a sum of all odd modules plus inputs
        # out0 = input + g1 + g2 + ... + gn  -- a sum of all even modules plus inputs
        # hence, out0 + out1 - inp1 = input + f1 + g1 + g2 + g2 + ... + fn + gn
        return out0 + out1 - inp1
