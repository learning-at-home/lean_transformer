from itertools import zip_longest
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.checkpoint import get_device_states, set_device_states

from lean_transformer.blocksparse.linear import GeneralizedLinear, _GeneralizedLinear, TritonMatmulForLinearLayer
from lean_transformer.utils import ACT2FN


class LeanFFN(nn.Module):
    """
    A transformer FFN module that doesn't hog your GPU memory. Uses a manually optimized differentiation algorithm.

    :param hidden_size: base hidden size of the transformer
    :param intermediate_size: a (typically larger) hidden dimension where activation is applied
    :param activation: a pytorch nonlinearity to use in the intermediate layer
    :param gated: use gated activations based on https://arxiv.org/abs/2002.05202 and https://arxiv.org/abs/2102.11972
      note: gated activations require 1.5x more parameters compared to their non-gated variants.
    :param layer_norm_eps: see torch.nn.functional.layer_norm
    :param post_layer_norm: if set, applies an additional layer norm to projected attention outputs before residuals,
       as proposed in the CogView paper ( arXiv:2105.13290 ). This is meant to make fp16 training
       more stable for deep transformers. This technique is also a part of NormFormer ( arXiv:2110.09456 )
    :param dropout: hidden dropout probability, applied to the output projection (before adding residual)
    :param residual: if True, adds the original layer input to the final layer output
    :param ffn_custom_grad: if True (default), use custom backprop code that saves memory at the cost of ~5% extra compute

    :param i2h_proj: custom *first* linear layer (hidden_size -> intermediate_size or 2x indermediate_size)
    :param h2o_proj: custom *second* linear layer (intermediate_size -> hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation=ACT2FN["gelu_fused"],
        gated: bool = False,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
        post_layer_norm: bool = False,
        i2h_proj: Optional[nn.Linear] = None,
        h2o_proj: Optional[nn.Linear] = None,
        residual: bool = True,
        ffn_custom_grad: bool = False,
    ):
        super().__init__()
        i2h_out_features = intermediate_size * 2 if gated else intermediate_size
        self.i2h_proj = nn.Linear(hidden_size, i2h_out_features) if i2h_proj is None else i2h_proj
        self.h2o_proj = nn.Linear(intermediate_size, hidden_size) if h2o_proj is None else h2o_proj
        ffn_custom_grad = False
        if ffn_custom_grad:
            assert type(self.i2h_proj) in (nn.Linear, GeneralizedLinear), "custom grad supports only nn.Linear and GeneralizedLinear"
            assert type(self.h2o_proj) in (nn.Linear, GeneralizedLinear), "custom grad supports only nn.Linear and GeneralizedLinear"
        assert self.i2h_proj.in_features == self.h2o_proj.out_features == hidden_size
        assert self.i2h_proj.out_features == i2h_out_features and self.h2o_proj.in_features == intermediate_size
        self.pre_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if post_layer_norm else None
        self.activation = activation
        self.gated = gated
        self.dropout = dropout
        self.residual = residual
        self.ffn_custom_grad = ffn_custom_grad

    def forward(self, input):
        assert not self.ffn_custom_grad
        return self._forward_custom(input) if self.ffn_custom_grad else self._forward_pytorch(input)

    def _forward_pytorch(self, input):
        input_2d = input.view(-1, input.shape[-1])
        input_ln = F.layer_norm(
            input_2d, input.shape[-1:], self.pre_layer_norm.weight, self.pre_layer_norm.bias, self.pre_layer_norm.eps
        )
        pre_activation = self.i2h_proj(input_ln)
        hid_act = _LeanFFN._apply_activation(pre_activation, self.activation, gated=self.gated)

        out = self.h2o_proj(hid_act)
        if self.post_layer_norm:
            out = self.post_layer_norm(out)

        if self.dropout and self.training:
            out = torch.dropout(out, self.dropout, self.training)
        if self.residual:
            out = out + input_2d
        return out.view(*input.shape)

    def _forward_custom(self, input):
        post_ln_weight = post_ln_bias = None
        if self.post_layer_norm is not None:
            post_ln_weight, post_ln_bias = self.post_layer_norm.weight, self.post_layer_norm.bias
        i2h_lowrank_first = i2h_lowrank_second = h2o_lowrank_first = h2o_lowrank_second = None
        i2h_forward_indices = i2h_backward_indices = h2o_forward_indices = h2o_backward_indices = None
        i2h_matmul_op = h2o_matmul_op = None
        if isinstance(self.i2h_proj, GeneralizedLinear):
            i2h_lowrank_first, i2h_lowrank_second = self.i2h_proj.get_combined_lowrank_components()
            i2h_forward_indices = self.i2h_proj.matrix.forward_indices
            i2h_backward_indices = self.i2h_proj.matrix.backward_indices
            i2h_matmul_op = self.i2h_proj.matrix.matmul_op
        if isinstance(self.h2o_proj, GeneralizedLinear):
            h2o_lowrank_first, h2o_lowrank_second = self.h2o_proj.get_combined_lowrank_components()
            h2o_forward_indices = self.h2o_proj.matrix.forward_indices
            h2o_backward_indices = self.h2o_proj.matrix.backward_indices
            h2o_matmul_op = self.h2o_proj.matrix.matmul_op

        output = _LeanFFN.apply(
            input,
            self.pre_layer_norm.weight,
            self.pre_layer_norm.bias,
            self.i2h_proj.weight,
            self.i2h_proj.bias,
            i2h_lowrank_first,
            i2h_lowrank_second,
            i2h_forward_indices,
            i2h_backward_indices,
            i2h_matmul_op,
            self.h2o_proj.weight,
            self.h2o_proj.bias,
            h2o_lowrank_first,
            h2o_lowrank_second,
            h2o_forward_indices,
            h2o_backward_indices,
            h2o_matmul_op,
            post_ln_weight,
            post_ln_bias,
            self.activation,
            self.gated,
            self.dropout,
            self.training,
            self.pre_layer_norm.eps,
            self.residual,
        )
        return output


class _LeanFFN(torch.autograd.Function):
    """Autograd function for transformer FFN, manually optimized to reduce memory without affecting performance"""

    @staticmethod
    def _apply_activation(pre_activation: torch.Tensor, activation: callable, gated: bool):
        if not gated:
            return activation(pre_activation)
        else:
            pre_gate, lin = pre_activation.split(pre_activation.shape[-1] // 2, dim=-1)
            return activation(pre_gate).mul_(lin)

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        i2h_weight: torch.Tensor,
        i2h_bias: Optional[torch.Tensor],
        i2h_lowrank_first: Optional[torch.Tensor],
        i2h_lowrank_second: Optional[torch.Tensor],
        i2h_forward_indices: Optional[torch.IntTensor],
        i2h_backward_indices: Optional[torch.IntTensor],
        i2h_matmul_op: Optional[TritonMatmulForLinearLayer],
        h2o_weight: torch.Tensor,
        h2o_bias: Optional[torch.Tensor],
        h2o_lowrank_first: Optional[torch.Tensor],
        h2o_lowrank_second: Optional[torch.Tensor],
        h2o_forward_indices: Optional[torch.IntTensor],
        h2o_backward_indices: Optional[torch.IntTensor],
        h2o_matmul_op: Optional[TritonMatmulForLinearLayer],
        post_ln_weight: Optional[torch.Tensor],
        post_ln_bias: Optional[torch.Tensor],
        activation: callable,
        gated: bool,
        dropout: float,
        training: bool,
        ln_eps: float,
        residual: bool,
    ):
        ctx._dropout, ctx._training, ctx._ln_eps = dropout, training, ln_eps
        ctx._activation, ctx._gated, ctx._residual = activation, gated, residual
        ctx._i2h_matmul_op, ctx._h2o_matmul_op = i2h_matmul_op, h2o_matmul_op
        ctx._use_post_ln = post_ln_weight is not None

        dropout_rng, post_ln_input = None, None  # optional tensors to save
        input_2d = input.view(-1, input.shape[-1])

        input_ln = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ln_eps)

        pre_activation, i2h_tensors = _GeneralizedLinear.forward_functional(
            input_ln, i2h_weight, i2h_bias, i2h_lowrank_first, i2h_lowrank_second,
            i2h_forward_indices, i2h_backward_indices, i2h_matmul_op
        )

        hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, ctx._gated)

        out, h2o_tensors = _GeneralizedLinear.forward_functional(
            hid_act, h2o_weight, h2o_bias, h2o_lowrank_first, h2o_lowrank_second,
            h2o_forward_indices, h2o_backward_indices, h2o_matmul_op
        )

        if ctx._use_post_ln:
            post_ln_input = out
            out = F.layer_norm(post_ln_input, post_ln_input.shape[-1:], post_ln_weight, post_ln_bias, eps=ln_eps)

        if training and dropout:
            dropout_rng = _LeanFFN._get_device_state(out)
            out = torch.dropout_(out, dropout, training)

        if residual:
            out = torch.add(out, input_2d, out=out if 'xla' not in out.device.type else None)

        assert i2h_tensors[0] is input_ln and h2o_tensors[0] is hid_act  # we can rematerialize these tensors
        tensors_to_save = [
            input, pre_activation, ln_weight, ln_bias, post_ln_input, post_ln_weight, post_ln_bias, dropout_rng
        ]
        tensors_to_save.extend((*i2h_tensors[1:], *h2o_tensors[1:]))
        ctx.save_for_backward(*tensors_to_save)
        ctx._num_i2h_tensors = len(i2h_tensors)
        ctx._num_h2o_tensors = len(h2o_tensors)
        return out.view(*input.shape)

    @staticmethod
    def _h2o_backward(ctx, grad_output: torch.Tensor, hid_act: torch.Tensor):
        saved_tensors = (hid_act, *ctx.saved_tensors[-ctx._num_h2o_tensors + 1 :])
        needs_input_grad = [hid_act.requires_grad, *ctx.needs_input_grad[10:17]]
        grads = _GeneralizedLinear.backward_functional(grad_output, saved_tensors, needs_input_grad, ctx._h2o_matmul_op)
        return tuple(grad if needed else None for grad, needed in zip_longest(grads, needs_input_grad))

    @staticmethod
    def _i2h_backward(ctx, grad_output: torch.Tensor, input_ln: torch.Tensor):
        saved_tensors = (input_ln, *ctx.saved_tensors[-ctx._num_i2h_tensors - ctx._num_h2o_tensors + 2 : -ctx._num_h2o_tensors + 1])
        needs_input_grad = [input_ln.requires_grad, *ctx.needs_input_grad[3:10]]
        grads = _GeneralizedLinear.backward_functional(grad_output, saved_tensors, needs_input_grad, ctx._i2h_matmul_op)
        return tuple(grad if needed else None for grad, needed in zip_longest(grads, needs_input_grad))

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_ln_weight = grad_ln_bias = grad_post_ln_weight = grad_post_ln_bias = None
        input, pre_activation, ln_weight, ln_bias, = ctx.saved_tensors[:4]
        post_ln_input, post_ln_weight, post_ln_bias, dropout_rng = ctx.saved_tensors[4: 8]
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

        # backward(... -> post_norm -> dropout -> residual)
        grad_residual_2d = grad_output_2d if ctx._residual else None
        if dropout_rng is not None:
            _LeanFFN._set_device_state(grad_output_2d, dropout_rng)
            grad_output_2d = torch.dropout(grad_output_2d, ctx._dropout, ctx._training)

        if ctx._use_post_ln:
            assert post_ln_input is not None
            with torch.enable_grad():
                required_grad = post_ln_input.requires_grad
                post_ln_input.requires_grad_(True)
                post_ln_out = F.layer_norm(
                    post_ln_input, post_ln_input.shape[-1:], post_ln_weight, post_ln_bias, eps=ctx._ln_eps
                )
                grad_output_2d, grad_post_ln_weight, grad_post_ln_bias = torch.autograd.grad(
                    post_ln_out, [post_ln_input, post_ln_weight, post_ln_bias], grad_outputs=grad_output_2d
                )
                post_ln_input.requires_grad_(required_grad)
                del post_ln_input, post_ln_out

        # backward(... -> nonlinearity -> linear_h2o -> ...)
        input_2d = input.view(-1, input.shape[-1])
        grad_h2o_output_2d = grad_output_2d.view(-1, grad_output.shape[-1])

        with torch.enable_grad():
            # rematerialize activation
            pre_activation.requires_grad_(True)
            hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, ctx._gated)

            with torch.no_grad():
                (grad_hid_act, grad_h2o_weight, grad_h2o_bias, grad_h2o_lowrank_first, grad_h2o_lowrank_second,
                 unused_grad_forward_indices, unused_grad_backward_indices, unused_grad_matmul_op) = \
                    _LeanFFN._h2o_backward(ctx, grad_h2o_output_2d, hid_act)

            (grad_hid,) = torch.autograd.grad(hid_act, pre_activation, grad_outputs=grad_hid_act)
            pre_activation.requires_grad_(False)
            del hid_act

        # backward(... -> input_layernorm -> linear_i2h -> ...)
        with torch.enable_grad():
            # rematerialize input_ln
            input_2d.requires_grad_(True)
            input_ln_2d = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ctx._ln_eps)

            with torch.no_grad():
                (grad_input_ln_2d, grad_i2h_weight, grad_i2h_bias, grad_i2h_lowrank_first, grad_i2h_lowrank_second,
                 unused_grad_forward_indices, unused_grad_backward_indices, unused_grad_matmul_op) = \
                    _LeanFFN._i2h_backward(ctx, grad_hid, input_ln_2d)

            if any(ctx.needs_input_grad[0:3]):
                partial_grad_input_2d, grad_ln_weight, grad_ln_bias = torch.autograd.grad(
                    outputs=input_ln_2d, inputs=[input_2d, ln_weight, ln_bias], grad_outputs=grad_input_ln_2d
                )
            del input_2d, input_ln_2d, grad_input_ln_2d

        # add up residual grads
        if ctx.needs_input_grad[0]:
            grad_input = partial_grad_input_2d
            if ctx._residual:
                grad_input = grad_input.add_(grad_residual_2d)
            grad_input = grad_input.view(*input.shape)

        return (grad_input, grad_ln_weight, grad_ln_bias,
                grad_i2h_weight, grad_i2h_bias, grad_i2h_lowrank_first, grad_i2h_lowrank_second, None, None, None,
                grad_h2o_weight, grad_h2o_bias, grad_h2o_lowrank_first, grad_h2o_lowrank_second, None, None, None,
                grad_post_ln_weight, grad_post_ln_bias, None, None, None, None, None, None)

    @staticmethod
    def _get_device_state(x: torch.Tensor):
        if x.device.type == 'cuda':
            _, (state, ) = get_device_states(x)
            return state
        else:
            return torch.get_rng_state()

    @staticmethod
    def _set_device_state(x: torch.Tensor, state: torch.Tensor):
        if x.device.type == 'cuda':
            return set_device_states([x.get_device()], [state])
        else:
            torch.set_rng_state(state)
