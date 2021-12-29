"""
A module that implements sequential model type with optional keyword arguments.
When using gradient checkpoints or reversible sequential, keyword arguments should NOT require grad.
"""
import contextlib
import copy
import typing

import torch
from torch import nn as nn
from torch.utils.checkpoint import checkpoint
from revlib import MemoryModes, ReversibleModuleCache, replace_grad, ReversibleModule
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
    CURRENT_KWARGS, KWARGS_FOR_BACKWARD, GRAD_ENABLED = None, dict(), True

    def __init__(self, module: nn.Module, active_keys=(), use_first_output: bool = False):
        super().__init__()
        self.module, self.active_keys, self.use_first_output = module, active_keys, use_first_output

    @contextlib.contextmanager
    @classmethod
    def using_kwargs(cls, kwargs, grad_enabled: bool):
        assert cls.CURRENT_KWARGS is None, "nesting is not supported"
        if cls.KWARGS_FOR_BACKWARD:
            logger.warning(
                "Not all kwargs from forward pass were used in backward pass. This is expected if you ran inference"
                " without zero_grad or forgot to run backward. Otherwise something terrible just happened.")

        cls.CURRENT_KWARGS, cls.KWARGS_FOR_BACKWARD, cls.GRAD_ENABLED = kwargs, dict(), grad_enabled
        try:
            yield
        finally:
            cls.CURRENT_KWARGS = None

    def get_kwargs(self):
        if id(self) in self.KWARGS_FOR_BACKWARD:
            assert self.CURRENT_KWARGS is None, "this code should only run during backward pass and outside the context"
            return self.KWARGS_FOR_BACKWARD.pop(id(self))
        else:
            active_kwargs = {k: v for k, v in self.CURRENT_KWARGS.items() if k in self.active_keys}
            assert id(self) not in self.KWARGS_FOR_BACKWARD, "Trying to run the same module twice"
            if self.GRAD_ENABLED:
                self.KWARGS_FOR_BACKWARD[id(self)] = active_kwargs
            return active_kwargs

    def forward(self, input: torch.Tensor):
        output = self.module(input, **self.get_kwargs())
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
        with ActiveKwargs.using_kwargs(kwargs, torch.is_grad_enabled() and input.requires_grad):
            for module in self:
                if self.gradient_checkpointing and torch.is_grad_enabled():
                    input = checkpoint(module, input)
                else:
                    input = module(input)
        return input


class ReversibleWithKwargs(torch.nn.Module):
    def __init__(self, *modules,
                 coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                 coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                 memory_mode: MemoryModes = MemoryModes.autograd_function,
                 target_device: str = ""):
        super().__init__()
        for module in modules:
            assert isinstance(module, ActiveKwargs)

        coupling_forward = list(coupling_forward) if coupling_forward else [None]
        coupling_inverse = list(coupling_inverse) if coupling_inverse else [None]
        memory_savings = memory_mode != MemoryModes.no_savings
        cache = ReversibleModuleCache() if memory_mode in (MemoryModes.checkpoint, MemoryModes.autograd_graph) else None
        self.replace_grad = replace_grad if memory_mode == MemoryModes.autograd_function else lambda *x: x
        self.stem = torch.nn.Sequential(*[m if isinstance(m, ReversibleModule) else ReversibleModule(
            m, coupling_forward[i % len(coupling_forward)], coupling_inverse[i % len(coupling_inverse)],
            memory_savings, copy.deepcopy(cache) if memory_mode == MemoryModes.checkpoint else cache, target_device)
                                          for i, m in enumerate(modules)])
        self.m = memory_mode

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        with ActiveKwargs.using_kwargs(kwargs, torch.is_grad_enabled() and input.requires_grad):
            inp1 = inp0 = input.to(torch.float32)  # enforce upcasting residuals to fp32
            zeros = torch.zeros_like(inp0)
            out0, out1 = self.replace_grad(*self.stem((inp0, inp1, zeros, zeros)))
            # note: we initialize both sides of reversible sequence with the same input (e.g. embeddings)
            # and combine them to get something resembling a normal residual sum. More specifically,
            # out1 = input + f1 + f2 + ... + fn  -- a sum of all odd modules plus inputs
            # out0 = input + g1 + g2 + ... + gn  -- a sum of all even modules plus inputs
            # hence, out0 + out1 - inp1 = input + f1 + g1 + g2 + g2 + ... + fn + gn
            return out0 + out1 - inp1


