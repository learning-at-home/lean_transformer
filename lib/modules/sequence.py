import contextlib
import copy
import typing

import torch
import torch.nn as nn
from torch import nn as nn
from torch.utils.checkpoint import checkpoint
from revlib import MemoryModes, ReversibleModuleCache, replace_grad, ReversibleModule, DUAL_OR_QUAD_TENSOR

CURRENT_KWARGS = None


@contextlib.contextmanager
def using_kwargs(kwargs):
    global CURRENT_KWARGS
    old, CURRENT_KWARGS = CURRENT_KWARGS, kwargs
    try:
        yield kwargs
    finally:
        CURRENT_KWARGS = old


def get_kwargs(active_keys):
    return {} if CURRENT_KWARGS is None else {k: v for k, v in CURRENT_KWARGS.items() if k in active_keys}


class ActiveKwargs(nn.Module):
    """Adapts a self-attention or ffn module to be a part of torch.nn.Sequential"""

    def __init__(self, module: nn.Module, active_keys=(), use_first_output: bool = False):
        super().__init__()
        self.module, self.active_keys, self.use_first_output = module, active_keys, use_first_output

    def forward(self, input: torch.Tensor):
        output = self.module(input, **get_kwargs(self.active_keys))
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
        with using_kwargs(kwargs):
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

    def forward(self, inp0: torch.Tensor, **kwargs) -> torch.Tensor:
        with using_kwargs(kwargs):
            inp0 = inp0.to(torch.float32)  # enforce upcasting residuals to fp32
            inp1 = zeros = torch.zeros_like(inp0)
            out0, out1 = self.replace_grad(*self.stem((inp0, inp1, zeros, zeros)))
            return out0 if len(self.stem) % 2 == 0 else out1  # return the last updated out


class GradientCheckpointingMixin:
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool):
        if isinstance(module, SequentialWithKwargs):
            module.gradient_checkpointing = value