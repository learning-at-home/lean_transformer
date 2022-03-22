from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from lean_transformer.utils import ACT2FN
from lean_transformer.ffn import LeanFFN
from lean_transformer.blocksparse.linear import GeneralizedLinear, GeneralizedMatrix

GELU = ACT2FN['gelu_fused']


class SimpleFFN(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, activation=GELU, layer_norm_eps=1e-12, dropout: float = 0.0
    ):
        super().__init__()
        self.i2h_proj = nn.Linear(hidden_size, intermediate_size)
        self.h2o_proj = nn.Linear(intermediate_size, hidden_size)
        self.pre_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.activation = activation
        self.dropout = dropout

    def forward(self, input):
        output = self.i2h_proj(self.pre_layer_norm(input))
        output = self.activation(output)
        output = self.h2o_proj(output)
        output = F.dropout(output, self.dropout)
        return output + input


@pytest.mark.forked
def test_ffn_simple():
    torch.use_deterministic_algorithms(True)

    batch_size = 4
    seq_len = 128
    dim = 32
    num_layers = 4

    baseline_ffn = SimpleFFN(dim, 4 * dim)
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


class ReferenceFFN(nn.Module):
    """
    A transformer FFN module that DOES hog your GPU memory.
    Complete with pre-LayerNorm, residual connections and dropout.

    :param gated: use gated activations based on https://arxiv.org/abs/2002.05202 and https://arxiv.org/abs/2102.11972
      note: gated activations require 1.5x more parameters compared to their non-gated variants.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation=GELU,
        gated: bool = False,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
        post_layer_norm: bool = False,
        i2h_proj: Optional[nn.Linear] = None,
        h2o_proj: Optional[nn.Linear] = None,
        residual: bool = True,
    ):
        super().__init__()
        i2h_out_features = intermediate_size * 2 if gated else intermediate_size
        self.i2h_proj = nn.Linear(hidden_size, i2h_out_features) if i2h_proj is None else i2h_proj
        self.h2o_proj = nn.Linear(intermediate_size, hidden_size) if h2o_proj is None else h2o_proj
        assert self.i2h_proj.in_features == self.h2o_proj.out_features == hidden_size
        assert self.i2h_proj.out_features == i2h_out_features and self.h2o_proj.in_features == intermediate_size
        self.pre_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if post_layer_norm else None
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

    def forward(self, input: torch.Tensor):
        input_2d = input.view(-1, input.shape[-1])
        input_ln = F.layer_norm(
            input_2d, input.shape[-1:], self.pre_layer_norm.weight, self.pre_layer_norm.bias, self.pre_layer_norm.eps
        )
        pre_activation = self.i2h_proj(input_ln)
        hid_act = self._apply_activation(pre_activation, self.activation, self.h2o_proj.in_features)

        out = self.h2o_proj(hid_act)
        if self.post_layer_norm:
            out = self.post_layer_norm(out)
        out = F.dropout(out, self.dropout, self.training)
        if self.residual:
            out = out + input_2d
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


@pytest.mark.parametrize(
    "adapter_dim,lowrank_dim,block_size,residual",
    [
        (0, 0, 0, True),
        (0, 0, 0, False),
        (0, 0, 16, False),
        (4, 0, 0, True),
        (0, 4, 0, True),
        (2, 6, 0, False),
        (6, 2, 16, True),
        (2, 2, 2, True),
        (1, 1, 1, False),
    ],
)
@pytest.mark.parametrize("custom_grad", [True, False])
@pytest.mark.forked
def test_ffn_shared(adapter_dim: int, lowrank_dim: int, block_size: int, residual: bool, custom_grad: bool):
    torch.use_deterministic_algorithms(True)

    batch_size = 4
    seq_len = 128
    dim = 32
    num_layers = 4
    block_size = 16
    layout = f"pixelfly({block_size})"
    rtol, atol = 1e-4, 1e-5
    baseline_ffn = ReferenceFFN(
        dim,
        4 * dim,
        gated=True,
        post_layer_norm=True,
        i2h_proj=GeneralizedLinear(GeneralizedMatrix(dim, 8 * dim, layout, lowrank_dim), adapter_dim),
        h2o_proj=GeneralizedLinear(GeneralizedMatrix(4 * dim, dim, layout, lowrank_dim), adapter_dim),
        residual=residual,
    )
    our_ffn = LeanFFN(
        dim,
        4 * dim,
        gated=True,
        post_layer_norm=True,
        i2h_proj=GeneralizedLinear(GeneralizedMatrix(dim, 8 * dim, layout, lowrank_dim), adapter_dim),
        h2o_proj=GeneralizedLinear(GeneralizedMatrix(4 * dim, dim, layout, lowrank_dim), adapter_dim),
        residual=residual,
        custom_grad=custom_grad,
    )
    with torch.no_grad():
        baseline_ffn.post_layer_norm.bias[...] = torch.randn_like(baseline_ffn.post_layer_norm.bias)
        baseline_ffn.post_layer_norm.weight[...] = torch.rand_like(baseline_ffn.post_layer_norm.weight) + 0.5

    assert our_ffn.load_state_dict(baseline_ffn.state_dict())

    x = torch.rand(batch_size, seq_len, dim, device="cpu", requires_grad=True)

    # test outputs
    out_ref = x
    for i in range(num_layers):
        out_ref = baseline_ffn.forward(out_ref)

    out_our = x
    for i in range(num_layers):
        out_our = our_ffn(out_our)

    assert torch.allclose(out_our, out_ref, rtol, atol)

    # test grad inputs
    obj = (out_ref * (out_ref + 1)).square().mean()
    (grad_ref,) = torch.autograd.grad(obj, x)

    obj = (out_our * (out_our + 1)).square().mean()
    (grad_our,) = torch.autograd.grad(obj, x)

    assert torch.allclose(grad_ref, grad_our, rtol, atol)

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
        assert torch.allclose(grad_ref, grad_our, rtol, atol)


@pytest.mark.forked
def test_ffn_dropout():
    torch.use_deterministic_algorithms(True)

    batch_size = 4
    seq_len = 128
    dim = 32
    num_layers = 4

    for dropout in (0.0, 0.5):
        our_ffn = LeanFFN(dim, 4 * dim, post_layer_norm=True, gated=True, dropout=dropout)
        x = torch.rand(batch_size, seq_len, dim, device="cpu", requires_grad=True)

        out = x
        for i in range(num_layers):
            out = our_ffn.forward(out)
        out.norm().backward()

        out1 = out
        grad1 = x.grad.clone()
        x.grad = None

        out = x
        for i in range(num_layers):
            out = our_ffn.forward(out)
        out.norm().backward()

        assert torch.allclose(out, out1, rtol=1e-3, atol=1e-5) == (dropout == 0)
        assert torch.allclose(x.grad, grad1, rtol=1e-3, atol=1e-5) == (dropout == 0)
