import pytest
import torch
import torch.nn.functional as F
from lean_transformer.utils import pad_to_multiple, GELU
import numpy as np


@pytest.mark.forked
def test_pad_to_multiple():
    x = torch.randn(3, 3)

    assert pad_to_multiple(x, multiple=3, dims=0) is x
    assert pad_to_multiple(x, multiple=3, dims=1) is x
    assert pad_to_multiple(x, multiple=2, dims=1) is not x
    assert pad_to_multiple(x, multiple=4, dims=1) is not x
    assert torch.allclose(pad_to_multiple(x, multiple=2, dims=1), pad_to_multiple(x, multiple=4, dims=1))
    assert pad_to_multiple(x, multiple=2, dims=0).shape == (4, 3)
    assert pad_to_multiple(x, multiple=4, dims=1).shape == (3, 4)
    assert pad_to_multiple(x, multiple=2, dims=[0, 1]).shape == (4, 4)
    assert pad_to_multiple(x, multiple=4, dims=1).sum().item() == x.sum().item()
    assert pad_to_multiple(x, multiple=10, dims=0)[3:].norm() == 0
    assert pad_to_multiple(x, multiple=4, dims=[0, 1]).shape == (4, 4)
    assert pad_to_multiple(x, multiple=3, dims=[0, 1]) is x

@pytest.mark.forked
def test_gelu():
    gelu_ours = GELU.apply(torch.linspace(-5, 5, 1000))
    gelu_ref = F.gelu(torch.linspace(-5, 5, 1000))
    assert abs(gelu_ours - gelu_ref).max().item() <= 5e-4