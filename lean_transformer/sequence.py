"""
A module that implements sequential model type with optional keyword arguments.
When using gradient checkpoints or reversible sequential, keyword arguments should NOT require grad.
"""
from contextlib import nullcontext
from typing import Sequence, Union

import torch

from lean_transformer.utils import get_logger
from revlib.core import ReversibleModule, ReversibleSequential
from revlib.utils import MomentumNetStem, MomentumNetSide
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
        self.gradient_checkpointing: Union[bool, int] = False
        self.checkpoint_last = False
        self.checkpoint_hook = None
        self.preserve_rng_state = True

    def forward(self, input: torch.Tensor, *args, **kwargs):
        assert int(self.gradient_checkpointing) >= 0, "gradient checkpointing must be either bool or a positive integer"
        kwarg_keys, kwarg_values = zip(*kwargs.items()) if kwargs else ([], [])
        use_checkpoints = self.gradient_checkpointing and torch.is_grad_enabled()

        depth = len(self)
        num_segments = depth if isinstance(self.gradient_checkpointing, bool) else int(self.gradient_checkpointing)
        segment_size = max(1, depth // num_segments)
        current_segment = []

        with self.checkpoint_hook if use_checkpoints and self.checkpoint_hook is not None else nullcontext():
            for i, module in enumerate(self):
                current_segment.append(module)
                if len(current_segment) == segment_size or i == depth - 1:
                    enabled = use_checkpoints and (i != depth - 1 or self.checkpoint_last)
                    # pack kwargs with args since gradient checkpoint does not support kwargs
                    if enabled:
                        input = checkpoint(self._run_modules, current_segment, input, kwarg_keys, *kwarg_values, *args,
                                           preserve_rng_state=self.preserve_rng_state)
                    else:
                        input = self._run_modules(current_segment, input, kwarg_keys, *kwarg_values, *args)
                    current_segment = []
        assert len(current_segment) == 0
        return input

    @staticmethod
    def _run_modules(modules: Sequence[ActiveKwargs], input: torch.Tensor, kwarg_keys: Sequence[str], *values):
        kwargs = {key: values[i] for i, key in enumerate(kwarg_keys)}
        args = values[len(kwarg_keys) :]
        for module in modules:
            input = module(input, *args, **kwargs)
        return input


class ReversibleWithKwargs(ReversibleSequential):
    def __init__(self, *modules, **kwargs):
        logger.warning("Using experimental AABB reorder!")
        modules_a, modules_b = [], []
        for i, m in enumerate(modules):
            (modules_a if i % 2 == 0 else modules_b).append(m)

        modules = []
        for i in range(len(modules_a) // 2):
            modules.extend(modules_a[2 * i: 2 * i + 2])
            modules.extend(modules_b[2 * i: 2 * i + 2])

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


class MomentumReversibleWithKwargs(ReversibleSequential):
    def __init__(self, *modules, beta: float, **kwargs):
        logger.warning("Current momentum net implementation is a hack, plz rewrite if it ends up working")
        momentum_modules = []
        for idx, module in enumerate(modules):
            assert isinstance(module, ActiveKwargs) or (
                    isinstance(module, ReversibleModule) and any(isinstance(m, ActiveKwargs) for m in module.modules())
            )
            assert not isinstance(module, ReversibleModule)
            assert type(module) == ActiveKwargs
            momentum_modules.append(
                ActiveKwargs(MomentumNetStem(module.module, beta ** idx), module.active_keys, module.use_first_output))
            momentum_modules.append(ActiveKwargs(MomentumNetSide((1 - beta) / beta ** (idx + 1)), ()))

        super().__init__(*momentum_modules, **kwargs)
        wrapped_modules = getattr(self, 'stem', [m for m in self.children() if isinstance(m, ReversibleModule)])
        assert len(wrapped_modules) == 2 * len(modules)
        self.stem = SequentialWithKwargs(*wrapped_modules)
        self.beta = beta

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        inp0 = input.to(torch.float32)  # enforce upcasting residuals to fp32
        inp1 = zeros = torch.zeros_like(inp0)
        out0, out1 = self.replace_grad(*self.stem((inp0, inp1, zeros, zeros), *args, **kwargs))
        return out0
