"""
This module implements weight matrix sharing for linear layer: full sharing and sharing with adapters
"""
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd


class SharedMatrix(nn.Module):
    """A module that stores a shared pytorch tensor for use in AdaptedLinear layers"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.matrix = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.matrix, std=math.sqrt(2.0 / (5 * min(out_features, in_features))))
        # note: the std is based on SmallInit (see https://arxiv.org/pdf/1910.05895.pdf section 2.2)

    @property
    def shape(self):
        return self.matrix.shape

    def __repr__(self):
        return f"{self.__class__.__name__}{tuple(self.matrix.shape)}"


class SharedLinear(nn.Linear):
    """A linear layer with a shared weight matrix and (optional) individual bias"""

    def __init__(self, shared_matrix: SharedMatrix, bias: bool = True):
        nn.Module.__init__(self)
        self.shared_matrix = shared_matrix
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

    @property
    def weight(self):
        return self.shared_matrix.matrix


class AdaptedLinear(SharedLinear):
    """A linear layer with a shared full-rank matrix and an individual low-rank adapter"""

    def __init__(self, shared_matrix: SharedMatrix, adapter_dim: int = 64, bias: bool = True):
        nn.Module.__init__(self)
        self.out_features, self.in_features = shared_matrix.shape
        self.shared_matrix = shared_matrix
        if adapter_dim != 0:
            self.adapter_first = nn.Parameter(torch.zeros(adapter_dim, self.in_features))
            self.adapter_second = nn.Parameter(torch.zeros(self.out_features, adapter_dim))
        else:
            self.adapter_first = self.adapter_second = None
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        # initialize in accordance with https://arxiv.org/pdf/2106.09685.pdf
        nn.init.xavier_normal_(self.adapter_first)
        nn.init.zeros_(self.adapter_second)

    def forward(self, input):
        return _GeneralizedLinear.apply(
            input, self.shared_matrix.matrix, self.bias, self.adapter_first, self.adapter_second
        )


class _GeneralizedLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        adapter_first: torch.Tensor,
        adapter_second: torch.Tensor,
    ):
        output, tensors_to_save = _GeneralizedLinear._forward_impl(input, weight, bias, adapter_first, adapter_second)
        ctx.save_for_backward(*tensors_to_save)
        return output

    @staticmethod
    def _forward_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        adapter_first: torch.Tensor,
        adapter_second: torch.Tensor,
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        adapter_hid = None
        output = F.linear(input, weight, bias)
        if adapter_first is not None:
            adapter_hid = F.linear(input, adapter_first)
            output = F.linear(adapter_hid, adapter_second, output)
        tensors_to_save = input, adapter_hid, weight, adapter_first, adapter_second
        return output, tensors_to_save

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        return _GeneralizedLinear._backward_impl(grad_output, *ctx.saved_tensors, needs_input_grad=ctx.needs_input_grad)

    @staticmethod
    def _backward_impl(grad_output: torch.Tensor, *saved_tensors: torch.Tensor, needs_input_grad: Sequence[bool]):
        grad_input = grad_weight = grad_adapter_first = grad_adapter_second = grad_bias = None
        input, adapter_hid, matrix, adapter_first, adapter_second = saved_tensors

        input_flat = input.flatten(0, -2)  # [etc, in_features]
        grad_output_flat = grad_output.flatten(0, -2)  # [etc, out_features]
        if adapter_hid is not None:
            adapter_hid_flat = adapter_hid.flatten(0, -2)  # [etc, adapter_dim]

        if adapter_first is not None and (needs_input_grad[0] or needs_input_grad[3]):
            grad_adapter_hid_flat = torch.matmul(grad_output_flat, adapter_second)  # [etc, adapter_dim]
        if needs_input_grad[1] or needs_input_grad[4]:
            grad_output_flat_transposed = grad_output_flat.t()  # [out_features, etc]

        if needs_input_grad[4]:
            grad_adapter_second = torch.matmul(grad_output_flat_transposed, adapter_hid_flat)
            # ^-- [out_features, adapter_dim]
        if needs_input_grad[3]:
            grad_adapter_hid_flat_transposed = grad_adapter_hid_flat.t()  # [adapter_dim, etc]
            grad_adapter_first = torch.matmul(grad_adapter_hid_flat_transposed, input_flat)
            # ^-- [adapter_dim, in_features]
        if needs_input_grad[2]:
            grad_bias = grad_output_flat.sum(dim=0)  # [out_features]
        if needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output_flat_transposed, input_flat)
            # ^-- [out_features, in_features]
        if needs_input_grad[0]:
            grad_input_flat = torch.matmul(grad_output_flat, matrix)
            if adapter_first is not None:
                if grad_input_flat.dtype == adapter_first.dtype == grad_adapter_hid_flat.dtype:
                    grad_input_flat = grad_input_flat.addmm_(grad_adapter_hid_flat, adapter_first)
                else:
                    grad_input_flat = torch.addmm(grad_input_flat, grad_adapter_hid_flat, adapter_first)
            grad_input = grad_input_flat.view_as(input)
        return grad_input, grad_weight, grad_bias, grad_adapter_first, grad_adapter_second
