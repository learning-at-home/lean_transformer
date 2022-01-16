import pytest
import torch
from torch import nn as nn
from torch.nn import functional as F

from lib import SemiSharedLinear, SharedMatrix


class ReferenceLinear(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.shared_matrix(input)
        if self.adapter_first is not None:
            output = F.linear(F.linear(input, self.adapter_first), self.adapter_second, output)
        if self.bias is not None:
            output += self.bias
        return output


@pytest.mark.parametrize("adapter_dim", [0, 4])
@pytest.mark.parametrize("lowrank_dim", [0, 60])
@pytest.mark.parametrize("block_size", [0, 8])
def test_linear(block_size: int, lowrank_dim: int, adapter_dim: int):
    torch.use_deterministic_algorithms(True)

    batch_size = 4
    dim = 128
    num_layers = 4

    baseline_ffn = ReferenceLinear(SharedMatrix(dim, dim, block_size, lowrank_dim), adapter_dim)
    our_ffn = SemiSharedLinear(SharedMatrix(dim, dim, block_size, lowrank_dim), adapter_dim)

    assert our_ffn.load_state_dict(baseline_ffn.state_dict())

    x = torch.rand(batch_size, dim, device="cpu", requires_grad=True)

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
        assert torch.allclose(grad_ref, grad_our)
