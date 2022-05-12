"""
This module implements weight matrix sharing for linear layer: full sharing and sharing with adapters
"""
import math
from itertools import zip_longest
from typing import Optional, Tuple, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from lean_transformer.blocksparse.layout import get_blocksparse_layout, get_indices_from_layout
from lean_transformer.blocksparse.native_backend import blocksparse_matmul, blocksparse_matmul_backward
from lean_transformer.blocksparse.triton_backend import TritonMatmulForLinearLayer, TRITON_PAD_TO
from lean_transformer.utils import maybe_script, pad_to_multiple

HID_SIZE = 2048
BLOCK_SIZE = 128
CODEBOOK_SIZE = 3 * 12 * (HID_SIZE // BLOCK_SIZE)


class GeneralizedMatrix(nn.Module):
    """A module that stores a shared pytorch tensor for use in GeneralizedLinear layers"""

    def __init__(self, blocksparse_layout: Optional[str] = None,
                 lowrank_dim: int = 0, blocksparse_backend: str = 'native'):
        super().__init__()
        assert lowrank_dim == 0 and blocksparse_layout is None
        self.blocksparse_backend = blocksparse_backend
        self._matmul_op = None

        if blocksparse_layout is None:
            assert blocksparse_backend == 'native', "triton is only used for block-sparse matrices"
            # fully-connected weight matrix
            self.codebook = nn.Parameter(torch.empty(CODEBOOK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            torch.nn.init.normal_(self.codebook, std=math.sqrt(2.0 / (5 * HID_SIZE)))
            # note: this is usually overwritten by the model-wide initialization
            self.layout = self.forward_indices = self.backward_indices = None
        else:
            raise NotImplementedError()
        self.lowrank_first = self.lowrank_second = None

    def make_weight(self, codebook_scales):
        # scales shape: [out_features // BLOCK_SIZE, in_features // BLOCK_SIZE, NUM_CODEBOOKS]
        weight_flat = (codebook_scales.flatten(0, 1) @ self.codebook.flatten(1, -1)
                       ).view(*codebook_scales.shape[:2], BLOCK_SIZE, BLOCK_SIZE).swapaxes_(1, 2)
        # [out_features // BLOCK_SIZE, BLOCK_SIZE, in_features // BLOCK_SIZE, BLOCK_SIZE]
        return weight_flat.flatten(2, -1).flatten(0, 1)

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
                output = self.matmul_op(input, self.weight)
        else:
            # dense weight
            output = F.linear(input, self.weight)

        if self.lowrank_first is not None and not ignore_lowrank:
            output = F.linear(F.linear(input, self.lowrank_first), self.lowrank_second, output)
        return output

    @property
    def matmul_op(self) -> Optional[TritonMatmulForLinearLayer]:
        if self.blocksparse_backend == 'native':
            return None
        if self._matmul_op is None:
            if self.weight.ndim != 4 or self.weight.shape[0] != 1 or self.weight.shape[-1] != self.weight.shape[-2]:
                raise ValueError("weights are not in triton format")
            block_size = self.weight.shape[-1]
            self._matmul_op = TritonMatmulForLinearLayer(self.layout, block_size)
        return self._matmul_op


class GeneralizedLinear(nn.Linear):
    """A linear layer with a shared full-rank matrix and an individual low-rank adapter"""

    def __init__(self, matrix: GeneralizedMatrix, out_features:int, in_features: int, adapter_dim: int = 0, bias: bool = True):
        nn.Module.__init__(self)
        assert adapter_dim != 0
        self.matrix = matrix
        self.out_features, self.in_features = out_features, in_features
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self.scale = nn.Parameter(torch.ones(self.out_features))
        assert min(out_features, in_features) == HID_SIZE
        self.codebook_scales = nn.Parameter(torch.rand(out_features // BLOCK_SIZE, in_features // BLOCK_SIZE, CODEBOOK_SIZE))
        with torch.no_grad():
            self.codebook_scales.data /= self.codebook_scales.sum(dim=-1, keepdim=True)  # idk if this matters

        if adapter_dim != 0:
            self.adapter_first = nn.Parameter(torch.zeros(adapter_dim, self.in_features))
            self.adapter_second = nn.Parameter(torch.zeros(self.out_features * 2, adapter_dim))

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
        return self.matrix.make_weight(self.codebook_scales)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_base = F.linear(input, self.weight)
        bias_or_zeros = self.bias if self.bias is not None else torch.zeros_like(self.scale)
        scale_and_bias = torch.cat([self.scale, bias_or_zeros], dim=0)
        hid = F.linear(input, self.adapter_first)
        multiplicative, additive = F.linear(hid, self.adapter_second, scale_and_bias).split(self.out_features, dim=-1)
        return torch.addcmul(additive, multiplicative, output_base)
        # return _GeneralizedLinear.apply(
        #     input, self.weight, self.bias, *self.get_combined_lowrank_components(),
        #     self.matrix.forward_indices, self.matrix.backward_indices, self.matrix.matmul_op)


class _GeneralizedLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, *args):
        output, tensors_to_save = _GeneralizedLinear.forward_functional(*args)
        ctx.save_for_backward(*tensors_to_save)
        ctx._matmul_op = args[-1]
        return output

    @staticmethod
    def forward_functional(
            input: torch.FloatTensor,
            main_weight: torch.FloatTensor,
            bias: Optional[torch.Tensor] = None,
            lowrank_first: Optional[torch.Tensor] = None,
            lowrank_second: Optional[torch.Tensor] = None,
            forward_indices: Optional[torch.Tensor] = None,
            backward_indices: Optional[torch.Tensor] = None,
            matmul_op: Optional[TritonMatmulForLinearLayer] = None
    ):
        """pure functional interface for use in other autograd functions"""
        if matmul_op is None:
            # matmul using pure pytorch, fused into _forward_jit
            extra_args = (forward_indices, backward_indices, None)
        else:
            # matmul using triton backend (or similar), can't be jit-compiled
            input_flat = input.flatten(0, -2)
            input_padded = pad_to_multiple(input_flat, TRITON_PAD_TO, dims=0)[None, None, ...]
            matmul_output, tensors_to_save = matmul_op.forward_functional(input_padded, main_weight)
            matmul_output = matmul_output.view(matmul_output.shape[2:])
            matmul_output = matmul_output[:len(input_flat)].view(*input.shape[:-1], -1)

            # check if we can re-materialize tensors during backward
            assert len(tensors_to_save) == 2
            assert tensors_to_save[0].data_ptr() == input_padded.data_ptr()
            assert tensors_to_save[1] is main_weight
            extra_args = (None, None, matmul_output.flatten(0, -2))

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
        matmul_output: Optional[torch.Tensor]
    ):
        input_flat = input.view(-1, input.shape[-1])
        if matmul_output is not None:
            output = matmul_output  # matmul was pre-computed in forward_functional
            if bias is not None:
                output.add_(bias.to(output.dtype))
        elif forward_indices is not None:
            # native sparse matmul
            output = blocksparse_matmul(input_flat, main_weight, forward_indices)
            if bias is not None:
                output.add_(bias.to(output.dtype))
        else:
            # native dense matmul
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
        return _GeneralizedLinear.backward_functional(grad_output, ctx.saved_tensors, ctx.needs_input_grad, ctx._matmul_op)

    @staticmethod
    def backward_functional(grad_output: torch.Tensor, saved_tensors: Sequence[torch.Tensor],
                            needs_input_grad: List[bool], matmul_op: Optional[TritonMatmulForLinearLayer] = None,
                            ) -> Sequence[Optional[torch.Tensor]]:
        """pure functional interface for use in other autograd functions"""
        if matmul_op is None:
            # matmul-backward using pure pytorch, fused into _backward_jit
            extra_args = (None, None)
        else:
            # matmul-backward using triton backend (or similar), can't be jit-compiled
            input, main_weight = saved_tensors[0], saved_tensors[2]
            matmul_needs_input_grad = (needs_input_grad[0], needs_input_grad[2])
            input_padded = pad_to_multiple(input.flatten(0, -2), TRITON_PAD_TO, dims=0)
            input_padded = input_padded.view(1, 1, *input_padded.shape)
            partial_grad_input, precomputed_grad_main_weight = matmul_op.backward_functional(
                grad_output, (input_padded, main_weight), matmul_needs_input_grad)
            extra_args = (partial_grad_input.flatten(0, -2), precomputed_grad_main_weight)

        grads = _GeneralizedLinear._backward_jit(
            grad_output, *saved_tensors, *extra_args, needs_input_grad=needs_input_grad
        )
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
                      partial_grad_input_from_matmul: Optional[torch.Tensor],
                      precomputed_grad_main_weight: Optional[torch.Tensor],
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

        # main matmul
        if partial_grad_input_from_matmul is not None and precomputed_grad_main_weight is not None:
            # matmul backward was pre-computed in _GeneralizedLinear.backward_functional
            grad_input_flat = partial_grad_input_from_matmul
            grad_main_weight = precomputed_grad_main_weight

        elif backward_indices is None:
            # dense shared matrix
            if needs_input_grad[1]:
                grad_main_weight = torch.matmul(grad_output_flat_transposed, input_flat)
                # ^-- [out_features, in_features]
            if needs_input_grad[0]:
                grad_input_flat = torch.matmul(grad_output_flat, main_weight)
        else:
            # block-sparse shared matrix
            if needs_input_grad[0] or needs_input_grad[1]:
                grad_input_flat, grad_main_weight = blocksparse_matmul_backward(
                    grad_output_flat, input_flat, main_weight, backward_indices,
                    input_requires_grad=needs_input_grad[0], weight_requires_grad=needs_input_grad[1])

        # low-rank adapter
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
