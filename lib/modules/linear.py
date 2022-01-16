"""
This module implements weight matrix sharing for linear layer: full sharing and sharing with adapters
"""
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from lib.modules.pixelfly import butterfly_matmul, get_butterfly_indices


class SharedMatrix(nn.Module):
    """A module that stores a shared pytorch tensor for use in SemiSharedLinear layers"""

    def __init__(self, in_features: int, out_features: int, block_size: int = 0, lowrank_dim: int = 0):
        super().__init__()
        self.out_features, self.in_features = out_features, in_features

        if block_size:
            # block-sparse weights with additive butterfly pattern
            forward_indices, backward_indices = get_butterfly_indices(
                out_features, in_features, block_size, stretch=False
            )
            self.register_buffer("forward_indices", forward_indices)
            self.register_buffer("backward_indices", backward_indices)
            active_blocks_per_input = self.forward_indices.numel() // (in_features // block_size)
            self.weight = nn.Parameter(torch.empty(in_features, active_blocks_per_input, block_size))
            nn.init.normal_(self.weight, std=math.sqrt(2.0 / (5 * min(out_features, in_features))))
            # note: the init std is based on SmallInit (see https://arxiv.org/pdf/1910.05895.pdf section 2.2)
            # it may be overwritten by model-level parameter "initializer_range"
        else:
            # fully-connected weight matrix
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.normal_(self.weight, std=math.sqrt(2. / (5 * min(out_features, in_features))))
            self.forward_indices = self.backward_indices = None

        if lowrank_dim:
            self.lowrank_first = nn.Parameter(torch.zeros(lowrank_dim, self.in_features))
            self.lowrank_second = nn.Parameter(torch.zeros(self.out_features, lowrank_dim))
            nn.init.normal_(self.lowrank_first, std=math.sqrt(2.0 / (5 * min(out_features, in_features))))
            nn.init.normal_(self.lowrank_second, std=math.sqrt(2.0 / (5 * min(out_features, in_features))))

        else:
            self.lowrank_first = self.lowrank_second = None

    @property
    def shape(self):
        return (self.out_features, self.in_features)

    def __repr__(self):
        return f"{self.__class__.__name__}{tuple(self.shape)}"

    def forward(self, input: torch.Tensor, *, ignore_lowrank: bool = False):
        """
        Multiply input tensor by this matrix with the same semantics as in torch.nn.Linear(..., bias=False)

        :param ignore_lowrank: if True, the low-rank components (lowrank_dim) will not be used in matrix multiplication
        """
        if self.forward_indices is not None:
            output = butterfly_matmul(input, self.weight, self.forward_indices)
        else:
            output = F.linear(input, self.weight)
        if self.lowrank_first is not None and not ignore_lowrank:
            output = F.linear(F.linear(input, self.lowrank_first), self.lowrank_second, output)
        return output


class SemiSharedLinear(nn.Linear):
    """A linear layer with a shared full-rank matrix and an individual low-rank adapter"""

    def __init__(self, shared_matrix: SharedMatrix, adapter_dim: int = 0, bias: bool = True):
        nn.Module.__init__(self)
        self.shared_matrix = shared_matrix
        self.out_features, self.in_features = self.shared_matrix.shape
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
        """Group together low-rank matrix components from this layer's adapter and SharedMatrix for faster matmul"""
        if self.adapter_first is not None and self.shared_matrix.lowrank_first is None:
            return self.adapter_first, self.adapter_second
        elif self.adapter_first is None and self.shared_matrix.lowrank_first is not None:
            return self.shared_matrix.lowrank_first, self.shared_matrix.lowrank_second
        elif self.adapter_first is not None and self.shared_matrix.lowrank_first is not None:
            combined_first = torch.cat([self.shared_matrix.lowrank_first, self.adapter_first], dim=0)
            # ^-- cat0[(lowrank_dim x input_dim), (adapter_dim, input_dim)] -> (combined_dim, input_dim)
            combined_second = torch.cat([self.shared_matrix.lowrank_second, self.adapter_second], dim=1)
            # ^-- cat1[(output_dim x lowrank_dim), (output_dim, adapter_dim)] -> (combined_dim, input_dim)
            return combined_first, combined_second
        else:
            assert self.adapter_first is None and self.adapter_second is None
            assert self.shared_matrix.lowrank_first is None and self.shared_matrix.lowrank_second is None
            return None, None

    @property
    def weight(self):
        return self.shared_matrix.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.shared_matrix(input, ignore_lowrank=True)  # we apply low-rank components manually below
        lowrank_first, lowrank_second = self.get_combined_lowrank_components()
        if lowrank_first is not None:
            output = F.linear(F.linear(input, lowrank_first), lowrank_second, output)
        if self.bias is not None:
            output += self.bias
        return output


class _GeneralizedLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, *args):
        output, tensors_to_save = _GeneralizedLinear._forward_impl(*args)
        ctx.save_for_backward(*tensors_to_save)
        return output

    @staticmethod
    def _forward_impl(
        input: torch.Tensor,
        main_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        lowrank_first: Optional[torch.Tensor],
        lowrank_second: Optional[torch.Tensor],
        forward_indices: Optional[torch.IntTensor],
        backward_indices: Optional[torch.IntTensor],
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        if forward_indices is not None:
            output = butterfly_matmul(input, main_weight, forward_indices)
            if bias is not None:
                output += bias
        else:
            output = F.linear(input, main_weight, bias)

        if lowrank_first is not None:
            lowrank_hid = F.linear(input, lowrank_first)
            output = F.linear(lowrank_hid, lowrank_second, output)
        else:
            lowrank_hid = None

        tensors_to_save = input, lowrank_hid, main_weight, lowrank_first, lowrank_second, backward_indices
        return output, tensors_to_save

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        return _GeneralizedLinear._backward_impl(grad_output, *ctx.saved_tensors, needs_input_grad=ctx.needs_input_grad)

    @staticmethod
    def _backward_impl(grad_output: torch.Tensor, *saved_tensors: torch.Tensor, needs_input_grad: Sequence[bool]):
        grad_input = grad_main_weight = grad_lowrank_first = grad_lowrank_second = grad_bias = None
        input, lowrank_hid, main_weight, lowrank_first, lowrank_second, backward_indices = saved_tensors

        input_flat = input.flatten(0, -2)  # [etc, in_features]
        grad_output_flat = grad_output.flatten(0, -2)  # [etc, out_features]

        # part 1: reshape tensors for computation
        if lowrank_hid is not None:
            lowrank_hid_flat = lowrank_hid.flatten(0, -2)  # [etc, lowrank_dim]
        if lowrank_first is not None and (needs_input_grad[0] or needs_input_grad[3]):
            grad_lowrank_hid_flat = torch.matmul(grad_output_flat, lowrank_second)  # [etc, lowrank_dim]
        if needs_input_grad[1] or needs_input_grad[4]:
            grad_output_flat_transposed = grad_output_flat.t()  # [out_features, etc]

        # part 2: grad w.r.t. low-rank components only
        if needs_input_grad[4]:
            grad_lowrank_second = torch.matmul(grad_output_flat_transposed, lowrank_hid_flat)
            # ^-- [out_features, lowrank_dim]
        if needs_input_grad[3]:
            grad_lowrank_hid_flat_transposed = grad_lowrank_hid_flat.t()  # [lowrank_dim, etc]
            grad_lowrank_first = torch.matmul(grad_lowrank_hid_flat_transposed, input_flat)
            # ^-- [lowrank_dim, in_features]
        if needs_input_grad[2]:
            grad_bias = grad_output_flat.sum(dim=0)  # [out_features]

        if needs_input_grad[1]:
            grad_main_weight = torch.matmul(grad_output_flat_transposed, input_flat)
            # ^-- [out_features, in_features]
        if needs_input_grad[0]:
            grad_input_flat = torch.matmul(grad_output_flat, main_weight)
            if lowrank_first is not None:
                if grad_input_flat.dtype == lowrank_first.dtype == grad_lowrank_hid_flat.dtype:
                    grad_input_flat = grad_input_flat.addmm_(grad_lowrank_hid_flat, lowrank_first)
                else:
                    grad_input_flat = torch.addmm(grad_input_flat, grad_lowrank_hid_flat, lowrank_first)
            grad_input = grad_input_flat.view_as(input)
        return grad_input, grad_main_weight, grad_bias, grad_lowrank_first, grad_lowrank_second
