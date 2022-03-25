from typing import Optional

import pytest
import torch
from lean_transformer.blocksparse.linear import GeneralizedLinear, GeneralizedMatrix, _GeneralizedLinear
from torch import nn as nn
from torch.nn import functional as F


def adapted_linear_naive(
    input: torch.Tensor,
    matrix: torch.Tensor,
    adapter_first: torch.Tensor,
    adapter_second: torch.Tensor,
    bias: Optional[torch.Tensor],
):
    adapter_hid = F.linear(input, adapter_first)
    shared_out = F.linear(input, matrix, bias)
    return adapter_hid @ adapter_second.t() + shared_out


@pytest.mark.forked
def test_semishared_linear_naive():
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    rtol, atol = 1e-3, 1e-5

    input = torch.randn(3, 15, 1024, requires_grad=True)
    weight = torch.randn(4096, 1024, requires_grad=True)
    adapter_first = torch.randn(64, 1024, requires_grad=True)
    adapter_second = torch.randn(4096, 64, requires_grad=True)
    bias = torch.randn(4096, requires_grad=True)
    random_dir = torch.randn(3, 15, 4096)

    out_ours = _GeneralizedLinear.apply(input, weight, bias, adapter_first, adapter_second, None, None)
    torch.sum(out_ours * random_dir).backward()
    grads_ours = tuple(tensor.grad.clone() for tensor in (input, weight, adapter_first, adapter_second, bias))

    for tensor in (input, weight, adapter_first, adapter_second, bias):
        tensor.grad = None

    out_ref = adapted_linear_naive(input, weight, adapter_first, adapter_second, bias)
    torch.sum(out_ref * random_dir).backward()

    grads_ref = tuple(tensor.grad.clone() for tensor in (input, weight, adapter_first, adapter_second, bias))

    assert torch.allclose(out_ours, out_ref, rtol, atol)

    for grad_ours, grad_ref in zip(grads_ours, grads_ref):
        assert torch.allclose(grad_ours, grad_ref, rtol, atol)


class ReferenceLinear(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.matrix(input)
        if self.adapter_first is not None:
            output = F.linear(F.linear(input, self.adapter_first), self.adapter_second, output)
        if self.bias is not None:
            output += self.bias
        return output


@pytest.mark.parametrize("adapter_dim", [0, 4])
@pytest.mark.parametrize("lowrank_dim", [0, 60])
@pytest.mark.parametrize("block_size", [0, 8])
@pytest.mark.forked
def test_linear(block_size: int, lowrank_dim: int, adapter_dim: int):
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    rtol, atol = 1e-3, 1e-6

    batch_size = 4
    dim = 128
    num_layers = 4
    layout = f"pixelfly(block_size={block_size})" if block_size else None

    baseline_ffn = ReferenceLinear(GeneralizedMatrix(dim, dim, layout, lowrank_dim), adapter_dim)
    our_ffn = GeneralizedLinear(GeneralizedMatrix(dim, dim, layout, lowrank_dim), adapter_dim)

    assert our_ffn.load_state_dict(baseline_ffn.state_dict())

    x = torch.rand(batch_size, dim, device="cpu", requires_grad=True)

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
    x = torch.rand(batch_size, dim, device="cpu", requires_grad=True)

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
