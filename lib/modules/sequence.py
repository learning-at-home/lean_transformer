"""
A module that implements sequential model type with optional keyword arguments.
When using gradient checkpoints or reversible sequential, keyword arguments should NOT require grad.
"""
import copy
from itertools import zip_longest
from typing import Sequence, Optional, List, Callable

import torch
from torch import nn as nn
from torch.utils.checkpoint import checkpoint
from revlib import MemoryModes, ReversibleModuleCache, replace_grad, ReversibleModule, ReversibleSequential
from hivemind.utils.logging import get_logger


logger = get_logger(__file__)


class ActiveKwargs(nn.Module):
    """
    A module with kwargs that is compatible with sequential
    Usage: ony use this as a part of SequentialWithKwargs or ReversibleWithKwargs

    What happens internally:
    - during forward pass, enter ActiveKwargs.using_kwargs(**kwargs) context
    - call each ActiveKwargs module at most once
    - during backward, it will reuse previously recorded kwargs

    """
    def __init__(self, module: nn.Module, active_keys=(), use_first_output: bool = False):
        super().__init__()
        self.module, self.active_keys, self.use_first_output = module, active_keys, use_first_output

    def forward(self, input: torch.Tensor, kwarg_keys: Sequence[str], *kwarg_values):
        kwargs = {key: value for key, value in zip_longest(kwarg_keys, kwarg_values) if key in self.active_keys}
        output = self.module(input, **kwargs)
        if self.use_first_output and not isinstance(output, torch.Tensor):
            output = output[0]
        return output


class SequentialWithKwargs(nn.Sequential):
    def __init__(self, *modules: ActiveKwargs):
        for module in modules:
            assert isinstance(module, ActiveKwargs)
        super().__init__(*modules)
        self.gradient_checkpointing = False

    def forward(self, input: torch.Tensor, **kwargs):
        kwarg_keys, kwarg_values = zip(*kwargs.items())
        for module in self:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                input = checkpoint(module, input, kwarg_keys, *kwarg_values)
            else:
                input = module(input, kwarg_keys, *kwarg_values)
        return input


class ReversibleWithKwargs(ReversibleSequential):
    def __init__(self, *modules, **kwargs):
        for module in modules:
            assert isinstance(module, ActiveKwargs) or \
                   (isinstance(module, ReversibleModule) and isinstance(module.wrapped_module, ActiveKwargs))
        super().__init__(*modules, **kwargs)
        self.stem = SequentialWithKwargs(*self.stem)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        inp1 = inp0 = input.to(torch.float32)  # enforce upcasting residuals to fp32
        zeros = torch.zeros_like(inp0)
        out0, out1 = self.replace_grad(*self.stem((inp0, inp1, zeros, zeros), **kwargs))
        # note: we initialize both sides of reversible sequence with the same input (e.g. embeddings)
        # and combine them to get something resembling a normal residual sum. More specifically,
        # out1 = input + f1 + f2 + ... + fn  -- a sum of all odd modules plus inputs
        # out0 = input + g1 + g2 + ... + gn  -- a sum of all even modules plus inputs
        # hence, out0 + out1 - inp1 = input + f1 + g1 + g2 + g2 + ... + fn + gn
        return out0 + out1 - inp1


