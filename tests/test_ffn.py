from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modules.ffn import LeanFFN
from lib.modules.linear import SemiSharedLinear, SharedMatrix


class ReferenceFFNSimple(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, activation=F.gelu, layer_norm_eps=1e-12, dropout: float = 0.0
    ):
        super().__init__()
        self.dense_i2h = nn.Linear(hidden_size, intermediate_size)
        self.dense_h2o = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.activation = activation
        self.dropout = dropout

    def forward(self, input):
        output = self.dense_i2h(self.layer_norm(input))
        output = self.activation(output)
        output = self.dense_h2o(output)
        output = F.dropout(output, self.dropout)
        return output + input


def test_ffn_simple():
    torch.use_deterministic_algorithms(True)

    batch_size = 4
    seq_len = 128
    dim = 32
    num_layers = 4

    baseline_ffn = ReferenceFFNSimple(dim, 4 * dim)
    our_ffn = LeanFFN(dim, 4 * dim)

    assert our_ffn.load_state_dict(baseline_ffn.state_dict())

    x = torch.rand(batch_size, seq_len, dim, device="cpu", requires_grad=True)

    # test outputs
    out_ref = x
    for i in range(num_layers):
        out_ref = baseline_ffn.forward(out_ref)

    out_our = x
    for i in range(num_layers):
        out_our = our_ffn(out_our)

    assert torch.allclose(out_our, out_ref)

    # test grad inputs
    obj = (out_ref * (out_ref + 1)).square().mean()
    (grad_ref,) = torch.autograd.grad(obj, x)

    obj = (out_our * (out_our + 1)).square().mean()
    (grad_our,) = torch.autograd.grad(obj, x)
    assert torch.allclose(grad_ref, grad_our)

    # test grad params
    x = torch.rand(batch_size, seq_len, dim, device="cpu", requires_grad=True)

    out_ref = x
    for i in range(num_layers):
        out_ref = baseline_ffn.forward(out_ref)

    out_our = x
    for i in range(num_layers):
        out_our = our_ffn(out_our)

    obj = (out_ref * (out_ref + 1)).square().mean()
    grad_params_ref = torch.autograd.grad(obj, list(baseline_ffn.parameters()))

    obj = (out_our * (out_our + 1)).square().mean()
    grad_params_our = torch.autograd.grad(obj, list(our_ffn.parameters()))

    for grad_ref, grad_our in zip(grad_params_ref, grad_params_our):
        assert torch.allclose(grad_ref, grad_our)

class ReferenceFFNShared(nn.Module):
    """
    A transformer FFN module that doesn't hog your GPU memory.
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
        i2h_adapter_first = i2h_adapter_second = h2o_adapter_first = h2o_adapter_second = None
        if isinstance(self.dense_i2h, SemiSharedLinear):
            i2h_adapter_first, i2h_adapter_second = self.dense_i2h.adapter_first, self.dense_i2h.adapter_second
        if isinstance(self.dense_h2o, SemiSharedLinear):
            h2o_adapter_first, h2o_adapter_second = self.dense_h2o.adapter_first, self.dense_h2o.adapter_second

        input_2d = input.view(-1, input.shape[-1])
        input_ln = F.layer_norm(
            input_2d, input.shape[-1:], self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps
        )
        pre_activation = self.linear_forward(
            input_ln, self.dense_i2h.weight, self.dense_i2h.bias, i2h_adapter_first, i2h_adapter_second
        )
        hid_act = self._apply_activation(pre_activation, self.activation, self.dense_h2o.weight.shape[1])

        out = self.linear_forward(
            hid_act, self.dense_h2o.weight, self.dense_h2o.bias, h2o_adapter_first, h2o_adapter_second
        )
        if self.sandwich_norm:
            out = self.sandwich_norm(out)
        out = F.dropout(out, self.dropout, self.training)
        if self.residual:
            out = out.add(input_2d)
        return out.view(*input.shape)

    @staticmethod
    def linear_forward(input, weight, bias, adapter_first, adapter_second):
        output = F.linear(input, weight, bias)
        if adapter_first is not None:
            adapter_hid = F.linear(input, adapter_first)
            output = F.linear(adapter_hid, adapter_second, output)
        return output

    @staticmethod
    def _apply_activation(pre_activation: torch.Tensor, activation: callable, hid_size: int):
        if pre_activation.shape[-1] == hid_size:
            return activation(pre_activation)
        elif pre_activation.shape[-1] == 2 * hid_size:
            pre_gate, lin = pre_activation.split(pre_activation.shape[-1] // 2, dim=-1)
            return activation(pre_gate).mul_(lin)
        else:
            raise RuntimeError("The output size of FFN layer must be either 1x or 2x the intermediate_size.")


def test_ffn_shared():

    torch.use_deterministic_algorithms(True)

    batch_size = 4
    seq_len = 128
    dim = 32
    num_layers = 4

    baseline_ffn = ReferenceFFNShared(
        dim,
        4 * dim,
        gated=True,
        sandwich_norm=True,
        dense_i2h=SemiSharedLinear(SharedMatrix(dim, 8 * dim)),
        dense_h2o=SemiSharedLinear(SharedMatrix(4 * dim, dim)),
    )
    our_ffn = LeanFFN(
        dim,
        4 * dim,
        gated=True,
        sandwich_norm=True,
        dense_i2h=SemiSharedLinear(SharedMatrix(dim, 8 * dim)),
        dense_h2o=SemiSharedLinear(SharedMatrix(4 * dim, dim)),
    )
    with torch.no_grad():
        baseline_ffn.sandwich_norm.bias[...] = torch.randn_like(baseline_ffn.sandwich_norm.bias)
        baseline_ffn.sandwich_norm.weight[...] = torch.rand_like(baseline_ffn.sandwich_norm.weight) + 0.5

    assert our_ffn.load_state_dict(baseline_ffn.state_dict())

    x = torch.rand(batch_size, seq_len, dim, device="cpu", requires_grad=True)

    # test outputs
    out_ref = x
    for i in range(num_layers):
        out_ref = baseline_ffn.forward(out_ref)

    out_our = x
    for i in range(num_layers):
        out_our = our_ffn(out_our)

    assert torch.allclose(out_our, out_ref)

    # test grad inputs
    obj = (out_ref * (out_ref + 1)).square().mean()
    (grad_ref,) = torch.autograd.grad(obj, x)

    obj = (out_our * (out_our + 1)).square().mean()
    (grad_our,) = torch.autograd.grad(obj, x)
    assert torch.allclose(grad_ref, grad_our)

    # test grad params
    x = torch.rand(batch_size, seq_len, dim, device="cpu", requires_grad=True)

    out_ref = x
    for i in range(num_layers):
        out_ref = baseline_ffn.forward(out_ref)

    out_our = x
    for i in range(num_layers):
        out_our = our_ffn(out_our)

    obj = (out_ref * (out_ref + 1)).square().mean()
    grad_params_ref = torch.autograd.grad(obj, list(baseline_ffn.parameters()))

    obj = (out_our * (out_our + 1)).square().mean()
    grad_params_our = torch.autograd.grad(obj, list(our_ffn.parameters()))

    for grad_ref, grad_our in zip(grad_params_ref, grad_params_our):
        assert torch.allclose(grad_ref, grad_our)

