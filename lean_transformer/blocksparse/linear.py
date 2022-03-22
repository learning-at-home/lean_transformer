"""
This module implements weight matrix sharing for linear layer: full sharing and sharing with adapters
"""
import math
from itertools import zip_longest
from typing import Optional, Tuple, List, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from lean_transformer.blocksparse.layout import get_blocksparse_layout, get_indices_from_layout
from lean_transformer.blocksparse.native_backend import blocksparse_matmul, blocksparse_matmul_backward
from lean_transformer.utils import maybe_script, pad_to_multiple

try:
    import triton.ops
except Exception as e:
    triton = e


class GeneralizedMatrix(nn.Module):
    """A module that stores a shared pytorch tensor for use in GeneralizedLinear layers"""

    def __init__(
            self, in_features: int, out_features: int, blocksparse_layout: Optional[str] = None, lowrank_dim: int = 0,
            use_triton: bool = False
    ):
        super().__init__()
        self.out_features, self.in_features = out_features, in_features

        if blocksparse_layout is None:
            assert not use_triton, "triton is only used for block-sparse matrices"
            # fully-connected weight matrix
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            # note: this is usually overwritten by the model-wide initialization
            self.layout = self.forward_indices = self.backward_indices = None
        else:
            # block-sparse weights with additive butterfly pattern
            layout = get_blocksparse_layout(out_features, in_features, blocksparse_layout)
            assert out_features / layout.shape[0] == in_features / layout.shape[1] == in_features // layout.shape[1]
            block_size = in_features // layout.shape[1]
            self.register_buffer("layout", layout, persistent=True)

            if use_triton:
                assert not isinstance(triton, Exception), f"triton is not available: {triton}"
                self.weight = nn.Parameter(torch.empty(1, self.layout.sum().item(), block_size, block_size))
                self.forward_indices = self.backward_indices = None
            else:
                # native backend
                forward_indices, backward_indices = get_indices_from_layout(layout)
                self.register_buffer("forward_indices", forward_indices, persistent=True)
                self.register_buffer("backward_indices", backward_indices, persistent=True)
                active_blocks_per_input = self.forward_indices.numel() // (in_features // block_size)
                self.weight = nn.Parameter(torch.empty(in_features, active_blocks_per_input, block_size))

            torch.nn.init.normal_(self.weight, std=math.sqrt(2.0 / (5 * min(out_features, in_features))))
            # note: this init is usually overwritten by the model-wide init

        if lowrank_dim:
            self.lowrank_first = nn.Parameter(torch.zeros(lowrank_dim, self.in_features))
            self.lowrank_second = nn.Parameter(torch.zeros(self.out_features, lowrank_dim))
            nn.init.normal_(self.lowrank_first, std=math.sqrt(2.0 / (5 * min(out_features, in_features))))
            nn.init.normal_(self.lowrank_second, std=math.sqrt(2.0 / (5 * min(out_features, in_features))))
        else:
            self.lowrank_first = self.lowrank_second = None

    @property
    def shape(self):
        return self.out_features, self.in_features

    def __repr__(self):
        return f"{self.__class__.__name__}{tuple(self.shape)}"

    def forward(self, input: torch.Tensor, *, ignore_lowrank: bool = False):
        """
        Multiply input tensor by this matrix with the same semantics as in torch.nn.Linear(..., bias=False)

        :param ignore_lowrank: if True, the low-rank components (lowrank_dim) will not be used in matrix multiplication
        """
        if self.layout is not None:
            if self.forward_indices is not None:
                output = blocksparse_matmul(input, self.weight, self.forward_indices)
            else:
                output = self._triton_matmul(input)
        else:
            # dense weight
            output = F.linear(input, self.weight)

        if self.lowrank_first is not None and not ignore_lowrank:
            output = F.linear(F.linear(input, self.lowrank_first), self.lowrank_second, output)
        return output

    def _triton_matmul(self, input: torch.Tensor) -> torch.FloatTensor:
        if self.triton_matmul_op is None:
            self.triton_matmul_op = triton.ops.blocksparse.matmul(
                self.layout.cpu(), self.weight.shape[-1], 'dds', trans_a=False, trans_b=True)
        input_flat = input.flatten(0, -2)
        input_padded = pad_to_multiple(input_flat, multiple=self.weight.shape[-1], dims=0)
        output_flat = self.triton_matmul_op(input_padded[None, None, ...], self.weight).flatten(0, -2)
        output = output_flat[:input_flat.shape[0]].view(input.shape[:-1], output_flat.shape[-1])
        return output


class GeneralizedLinear(nn.Linear):
    """A linear layer with a shared full-rank matrix and an individual low-rank adapter"""

    def __init__(self, matrix: GeneralizedMatrix, adapter_dim: int = 0, bias: bool = True):
        nn.Module.__init__(self)
        self.matrix = matrix
        self.out_features, self.in_features = self.matrix.shape
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        if adapter_dim != 0:
            self.adapter_first = nn.Parameter(torch.zeros(adapter_dim, self.in_features))
            self.adapter_second = nn.Parameter(torch.zeros(self.out_features, adapter_dim))

            # initialize in accordance with https://arxiv.org/pdf/2106.09685.pdf
            nn.init.xavier_normal_(self.adapter_first)
            nn.init.zeros_(self.adapter_second)
        else:
            self.adapter_first = self.adapter_second = None

    def get_combined_lowrank_components(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Group together low-rank matrices from this layer's adapter and GeneralizedMatrix for faster matmul"""
        if self.adapter_first is not None and self.matrix.lowrank_first is None:
            return self.adapter_first, self.adapter_second
        elif self.adapter_first is None and self.matrix.lowrank_first is not None:
            return self.matrix.lowrank_first, self.matrix.lowrank_second
        elif self.adapter_first is not None and self.matrix.lowrank_first is not None:
            combined_first = torch.cat([self.matrix.lowrank_first, self.adapter_first], dim=0)
            # ^-- cat0[(lowrank_dim x input_dim), (adapter_dim, input_dim)] -> (combined_dim, input_dim)
            combined_second = torch.cat([self.matrix.lowrank_second, self.adapter_second], dim=1)
            # ^-- cat1[(output_dim x lowrank_dim), (output_dim, adapter_dim)] -> (combined_dim, input_dim)
            return combined_first, combined_second
        else:
            assert self.adapter_first is None and self.adapter_second is None
            assert self.matrix.lowrank_first is None and self.matrix.lowrank_second is None
            return None, None

    @property
    def weight(self):
        return self.matrix.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _GeneralizedLinear.apply(
            input, self.weight, self.bias, *self.get_combined_lowrank_components(),
            self.matrix.forward_indices, self.matrix.backward_indices)


class _GeneralizedLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, *args):
        output, tensors_to_save = _GeneralizedLinear.forward_functional(*args)
        ctx.save_for_backward(*tensors_to_save)
        return output

    @staticmethod
    def forward_functional(input: torch.FloatTensor,
                           main_weight: torch.FloatTensor,
                           bias: Optional[torch.Tensor] = None,
                           lowrank_first: Optional[torch.Tensor] = None,
                           lowrank_second: Optional[torch.Tensor] = None,
                           forward_indices: Optional[torch.Tensor] = None,
                           backward_indices: Optional[torch.Tensor] = None,
                           matmul_op: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
                           ):
        """pure functional interface for use in other autograd functions"""
        if matmul_op is None:
            # matmul using pure pytorch, fused into _forward_jit
            extra_args = (forward_indices, backward_indices, None)
        else:
            # matmul using triton backend (or similar), can't be jit-compiled
            partial_output = matmul_op(input, main_weight)
            extra_args = (None, None, partial_output)

        output, *tensors_to_save = _GeneralizedLinear._forward_jit(
            input, main_weight, bias, lowrank_first, lowrank_second, *extra_args
        )
        return output, tensors_to_save

    @staticmethod
    @maybe_script
    def _forward_jit(
        input: torch.Tensor,
        main_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        lowrank_first: Optional[torch.Tensor],
        lowrank_second: Optional[torch.Tensor],
        forward_indices: Optional[torch.Tensor],
        backward_indices: Optional[torch.Tensor],
        partial_output: Optional[torch.Tensor]
    ):
        input_flat = input.view(-1, input.shape[-1])
        if partial_output is not None:
            output = partial_output  # matmul was pre-computed in forward_functional
        elif forward_indices is not None:
            output = blocksparse_matmul(input_flat, main_weight, forward_indices)
            if bias is not None:
                output.add_(bias.to(output.dtype))
        else:
            output = F.linear(input_flat, main_weight, bias)

        if lowrank_first is not None and lowrank_second is not None:
            lowrank_hid = F.linear(input_flat, lowrank_first)
            if "xla" in output.device.type:  # xla does not support in-place ops
                output = torch.addmm(output, lowrank_hid, lowrank_second.t().to(output.dtype))
            else:
                output = torch.addmm(output, lowrank_hid, lowrank_second.t().to(output.dtype), out=output)
        else:
            lowrank_hid = None
        output = output.view(input.shape[:-1] + output.shape[-1:])
        return output, input, lowrank_hid, main_weight, lowrank_first, lowrank_second, backward_indices

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        return _GeneralizedLinear.backward_functional(grad_output, ctx.saved_tensors, ctx.needs_input_grad)

    @staticmethod
    def backward_functional(
            grad_output: torch.Tensor, saved_tensors: Sequence[torch.Tensor], needs_input_grad: List[bool]
            ) -> Sequence[Optional[torch.Tensor]]:
        """pure functional interface for use in other autograd functions"""
        grads = _GeneralizedLinear._backward_jit(grad_output, *saved_tensors, needs_input_grad=needs_input_grad)
        return tuple(grad if needed else None for grad, needed in zip_longest(grads, needs_input_grad))

    @staticmethod
    @maybe_script
    def _backward_jit(grad_output: torch.Tensor,
                       input: torch.Tensor,
                       lowrank_hid: Optional[torch.Tensor],
                       main_weight: torch.Tensor,
                       lowrank_first: Optional[torch.Tensor],
                       lowrank_second: Optional[torch.Tensor],
                       backward_indices: Optional[torch.Tensor],
                       needs_input_grad: List[bool]):
        grad_input = grad_input_flat = grad_main_weight = grad_lowrank_first = grad_lowrank_second = grad_bias \
            = grad_output_flat_transposed = grad_lowrank_hid_flat = lowrank_hid_flat = torch.empty(0)
        input_flat = input.flatten(0, -2)  # [etc, in_features]
        grad_output_flat = grad_output.flatten(0, -2)  # [etc, out_features]

        if lowrank_hid is not None:
            lowrank_hid_flat = lowrank_hid.flatten(0, -2)  # [etc, lowrank_dim]
        if lowrank_first is not None and (needs_input_grad[0] or needs_input_grad[3]):
            assert lowrank_second is not None
            grad_lowrank_hid_flat = torch.matmul(grad_output_flat, lowrank_second)  # [etc, lowrank_dim]
        if needs_input_grad[1] or needs_input_grad[4]:
            grad_output_flat_transposed = grad_output_flat.t()  # [out_features, etc]

        if needs_input_grad[4]:
            assert lowrank_second is not None
            grad_lowrank_second = torch.matmul(grad_output_flat_transposed, lowrank_hid_flat)
            # ^-- [out_features, lowrank_dim]
        if needs_input_grad[3]:
            grad_lowrank_hid_flat_transposed = grad_lowrank_hid_flat.t()  # [lowrank_dim, etc]
            grad_lowrank_first = torch.matmul(grad_lowrank_hid_flat_transposed, input_flat)
            # ^-- [lowrank_dim, in_features]
        if needs_input_grad[2]:
            grad_bias = grad_output_flat.sum(dim=0)  # [out_features]
        if backward_indices is None:
            # dense shared matrix
            if needs_input_grad[1]:
                grad_main_weight = torch.matmul(grad_output_flat_transposed, input_flat)
                # ^-- [out_features, in_features]
            if needs_input_grad[0]:
                grad_input_flat = torch.matmul(grad_output_flat, main_weight)
        else:
            # block-sparse shared matrix
            grad_input_flat, grad_main_weight = blocksparse_matmul_backward(
                grad_output_flat, input_flat, main_weight, backward_indices,
                input_requires_grad=needs_input_grad[0], weight_requires_grad=needs_input_grad[1])

        if needs_input_grad[0] and lowrank_first is not None:
            # grad w.r.t. input through low-rank components
            if 'xla' not in grad_output.device.type:
                grad_input_flat = grad_input_flat.addmm_(
                    grad_lowrank_hid_flat.to(grad_output_flat.dtype),
                    lowrank_first.to(grad_output_flat.dtype)
                )
            else:
                grad_input_flat = torch.addmm(grad_input_flat, grad_lowrank_hid_flat, lowrank_first)
        if needs_input_grad[0]:
            grad_input = grad_input_flat.view_as(input)
        return grad_input, grad_main_weight, grad_bias, grad_lowrank_first, grad_lowrank_second, None, None
