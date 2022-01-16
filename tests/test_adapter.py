from typing import Optional

import torch
import torch.nn.functional as F

from lib.modules.linear import _SemiSharedLinear


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


def test_adapter_forward_backward():
    input = torch.randn(3, 15, 1024, requires_grad=True)
    weight = torch.randn(4096, 1024, requires_grad=True)
    adapter_first = torch.randn(64, 1024, requires_grad=True)
    adapter_second = torch.randn(4096, 64, requires_grad=True)
    bias = torch.randn(4096, requires_grad=True)
    random_dir = torch.randn(3, 15, 4096)

    out_ours = _SemiSharedLinear.apply(input, weight, bias, adapter_first, adapter_second, None, None)
    torch.sum(out_ours * random_dir).backward()
    grads_ours = tuple(tensor.grad.clone() for tensor in (input, weight, adapter_first, adapter_second, bias))

    for tensor in (input, weight, adapter_first, adapter_second, bias):
        tensor.grad = None

    out_ref = adapted_linear_naive(input, weight, adapter_first, adapter_second, bias)
    torch.sum(out_ref * random_dir).backward()

    grads_ref = tuple(tensor.grad.clone() for tensor in (input, weight, adapter_first, adapter_second, bias))

    assert torch.allclose(out_ours, out_ref)

    for grad_ours, grad_ref in zip(grads_ours, grads_ref):
        assert torch.allclose(grad_ours, grad_ref)
