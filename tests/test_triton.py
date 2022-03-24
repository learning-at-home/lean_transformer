import pytest
import torch
from lean_transformer.blocksparse import GeneralizedLinear, GeneralizedMatrix


@pytest.mark.forked
def test_triton_linear():
    if not torch.cuda.is_available():
        pytest.skip("This test requires GPU")
    layer = GeneralizedLinear(
        GeneralizedMatrix(512, 1536, blocksparse_layout="pixelfly(32)", use_triton=True, lowrank_dim=32)
    ).cuda()

    with torch.no_grad():
        layer.bias[...] = torch.linspace(-10, 10, len(layer.bias)).cuda()

    input = torch.randn(3, 14, 512).cuda()
    out = layer(input)

    assert torch.allclose(out, layer.matrix(input) + layer.bias, rtol=0, atol=1e-6)
