import functools
import os

import torch
from transformers.activations import ACT2FN as HF_ACT2FN


@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    if os.environ.get("TPU_NAME"):
        # this is a reserved variable that must be set to TPU address (e.g. grpc://11.22.33.44:1337) for TPU to function
        return fn
    else:
        return torch.jit.script(fn)


@maybe_script
def gelu_fwd(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@maybe_script
def gelu_back(grad_output, x):
    tanh = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    jac = 0.5 * x * ((1 - tanh * tanh) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh)
    return jac * grad_output


class _FusedGeLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        tmp = gelu_back(grad_output, input)
        return tmp


ACT2FN = dict(HF_ACT2FN, gelu_fused=_FusedGeLU.apply)
