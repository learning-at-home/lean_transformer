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


class GeneralizedMatrix(nn.Module):
    """A module that stores a shared pytorch tensor for use in GeneralizedLinear layers"""

    def __init__(self, in_features: int, out_features: int, blocksparse_layout: Optional[str] = None,
                 lowrank_dim: int = 0, blocksparse_backend: str = 'native'):
        super().__init__()
        assert lowrank_dim == 0
        self.out_features, self.in_features = out_features, in_features
        self.blocksparse_backend = blocksparse_backend
        self._matmul_op = None

        if blocksparse_layout is None:
            assert blocksparse_backend == 'native', "triton is only used for block-sparse matrices"
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

            if blocksparse_backend == 'native':
                # native backend
                forward_indices, backward_indices = get_indices_from_layout(layout)
                self.register_buffer("forward_indices", forward_indices, persistent=True)
                self.register_buffer("backward_indices", backward_indices, persistent=True)
                active_blocks_per_input = self.forward_indices.numel() // (in_features // block_size)
                self.weight = nn.Parameter(torch.empty(in_features, active_blocks_per_input, block_size))
            elif blocksparse_backend == 'triton':
                self.weight = nn.Parameter(torch.empty(1, self.layout.sum().item(), block_size, block_size))
                self.forward_indices = self.backward_indices = None
                # note: we do not initialize triton op until forward pass in case user decides to change layout
            else:
                raise NotImplementedError(f"Unrecognized sparsity backend: {blocksparse_backend}")

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

    def __init__(self, matrix: GeneralizedMatrix, adapter_dim: int = 0, bias: bool = True):
        nn.Module.__init__(self)
        assert adapter_dim == 0
        self.matrix = matrix
        self.out_features, self.in_features = self.matrix.shape
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self.scale = nn.Parameter(torch.ones(self.out_features))
        self.adapter_zlo = ZloebuchyAdapter(self.out_features, self.in_features)


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
        return self.matrix.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_base = F.linear(input, self.adapter_zlo(self.matrix.weight))
        bias_or_zeros = self.bias if self.bias is not None else torch.zeros_like(self.scale)
        return torch.addcmul(bias_or_zeros, self.scale, output_base)
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


BLOCK_SIZE = 256
TILE_SIZE = 4
RNG_HID = 96
RNG_POS_EMB = 8


class ZloebuchyAdapter(torch.nn.Module):
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.out_features, self.in_features = out_features, in_features
        self.out_blocks, self.in_blocks = out_features // BLOCK_SIZE, in_features // BLOCK_SIZE
        self.num_blocks = self.out_blocks * self.in_blocks
        self.w1 = nn.Parameter(
            torch.randn(self.num_blocks, TILE_SIZE ** 2, RNG_HID) * (2 / RNG_HID) ** 0.5
        )
        self.b1 = nn.Parameter(torch.zeros(self.num_blocks, RNG_HID))
        self.emb_h = nn.Parameter(torch.randn(self.num_blocks, BLOCK_SIZE // TILE_SIZE, RNG_POS_EMB) * 0.1)
        self.emb_w = nn.Parameter(torch.randn(self.num_blocks, BLOCK_SIZE // TILE_SIZE, RNG_POS_EMB) * 0.1)
        shared_matrix_std = (2 / (5 * min(out_features, out_features))) ** 0.5
        self.w2 = nn.Parameter(
            torch.randn(self.num_blocks, RNG_HID, TILE_SIZE ** 2) * (2 / TILE_SIZE ** 2) ** 0.5 * shared_matrix_std
        )
        self.b2 = nn.Parameter(torch.zeros(self.num_blocks, TILE_SIZE ** 2))
        self.scales = nn.Parameter(torch.ones(self.num_blocks, TILE_SIZE ** 2))

    @staticmethod
    @torch.jit.script
    def _forward_script(shared_matrix, w1, b1, emb_h, emb_w, w2, b2, scales,
                        TILE_SIZE: int, BLOCK_SIZE: int, RNG_HID: int, RNG_POS_EMB: int):
        out_blocks, in_blocks = shared_matrix.shape[0] // BLOCK_SIZE, shared_matrix.shape[1] // BLOCK_SIZE
        tiles_per_block = BLOCK_SIZE // TILE_SIZE
        shared_matrix_blocks = shared_matrix.view(
            out_blocks, tiles_per_block, TILE_SIZE,
            in_blocks, tiles_per_block, TILE_SIZE
        ).permute(0, 3, 1, 4, 2, 5).contiguous()
        # [out_blocks, in_blocks, tiles_per_block_h, tiles_per_block_w, tile_size, tile_size]

        shared_matrix_blocks_flat = shared_matrix_blocks.view(
            out_blocks * in_blocks, tiles_per_block * tiles_per_block, TILE_SIZE * TILE_SIZE)

        rng_hid = torch.bmm(shared_matrix_blocks_flat, w1).view(
            out_blocks * in_blocks, BLOCK_SIZE // TILE_SIZE, BLOCK_SIZE // TILE_SIZE, RNG_HID)

        rng_hid[:, :, :, :RNG_POS_EMB].add_(emb_h[:, :, None, :])
        rng_hid[:, :, :, RNG_POS_EMB: 2 * RNG_POS_EMB].add_(emb_w[:, None, :, :])
        rng_hid.add_(b1[:, None, None, :])
        rng_hid = F.elu_(rng_hid)

        rng_out = torch.bmm(rng_hid.flatten(1, 2), w2).add_(b2[:, None, :])
        rng_out = rng_out.addcmul_(
            shared_matrix_blocks_flat.to(rng_out.dtype), scales[:, None, :].to(rng_out.dtype)).view_as(
            shared_matrix_blocks)

        rng_out = rng_out.permute(0, 2, 4, 1, 3, 5).reshape_as(shared_matrix)
        return rng_out

    def forward(self, shared_matrix):
        return self._forward_script(
            shared_matrix, self.w1, self.b1, self.emb_h, self.emb_w, self.w2, self.b2, self.scales,
            TILE_SIZE, BLOCK_SIZE, RNG_HID, RNG_POS_EMB)
