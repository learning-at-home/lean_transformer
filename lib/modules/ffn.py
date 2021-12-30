from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from lib.modules.linear import AdaptedLinear, _GeneralizedLinear


class LeanFFN(nn.Module):
    """
    A transformer FFN module that DOES hog your GPU memory. TODO MAKE IT EFFICIENT AGAIN
    Complete with pre-LayerNorm, residual connections and dropout.

    :param gated: use gated activations based on https://arxiv.org/abs/2002.05202 and https://arxiv.org/abs/2102.11972
      note: gated activations require 1.5x more parameters compared to their non-gated variants.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation=F.gelu,
        gated: bool = False,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
        sandwich_norm: bool = False,
        dense_i2h: Optional[nn.Linear] = None,
        dense_h2o: Optional[nn.Linear] = None,
        residual: bool = True
    ):
        super().__init__()
        i2h_out_features = intermediate_size * 2 if gated else intermediate_size
        self.dense_i2h = nn.Linear(hidden_size, i2h_out_features) if dense_i2h is None else dense_i2h
        self.dense_h2o = nn.Linear(intermediate_size, hidden_size) if dense_h2o is None else dense_h2o
        assert self.dense_i2h.in_features == self.dense_h2o.out_features == hidden_size
        assert self.dense_i2h.out_features == i2h_out_features and self.dense_h2o.in_features == intermediate_size
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.sandwich_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if sandwich_norm else None
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

    def forward(self, input):
        input_2d = input.view(-1, input.shape[-1])
        input_ln = F.layer_norm(
            input_2d, input.shape[-1:], self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps
        )
        pre_activation = self.dense_i2h(input_ln)
        hid_act = self._apply_activation(pre_activation, self.activation, self.dense_h2o.weight.shape[1])

        out = self.dense_h2o(hid_act)
        if self.sandwich_norm:
            out = self.sandwich_norm(out)
        out = F.dropout(out, self.dropout, self.training)
        if self.residual:
            out = out.add(input_2d)
        return out.view(*input.shape)

    @staticmethod
    def _apply_activation(pre_activation: torch.Tensor, activation: callable, hid_size: int):
        if pre_activation.shape[-1] == hid_size:
            return activation(pre_activation)
        elif pre_activation.shape[-1] == 2 * hid_size:
            pre_gate, lin = pre_activation.split(pre_activation.shape[-1] // 2, dim=-1)
            return activation(pre_gate).mul_(lin)
        else:
            raise RuntimeError("The output size of FFN layer must be either 1x or 2x the intermediate_size.")

