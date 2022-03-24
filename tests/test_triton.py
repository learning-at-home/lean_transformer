import pytest
import torch
from lean_transformer.blocksparse import GeneralizedLinear, GeneralizedMatrix
from lean_transformer.blocksparse import TritonMatmulForLinearLayer, get_blocksparse_layout


def test_triton_blocksparse_op():
    if not torch.cuda.is_available():
        pytest.skip("This test requires GPU")

    op = TritonMatmulForLinearLayer(get_blocksparse_layout(128, 256, "pixelfly(32)"), block_size=32)
    weight = torch.randn(1, op.layout.sum(), 32, 32, device='cuda', requires_grad=True)
    input = torch.rand(14, 3, 256, requires_grad=True, device='cuda')

    out_naive = op(input, weight)

    out_functional, tensors_to_save = op.forward_functional(input, weight)
    grad_output = torch.randn_like(out_naive)
    grad_input_functional, grad_weight_functional = op.backward_functional(
        grad_output, tensors_to_save, (True, True))
    out_naive.backward(grad_output)

    assert torch.allclose(out_naive, out_functional, rtol=0, atol=1e-6)
    assert torch.allclose(input.grad, grad_input_functional, rtol=0, atol=1e-6)
    assert torch.allclose(weight.grad, grad_weight_functional, rtol=0, atol=1e-6)


@pytest.mark.forked
def test_triton_linear():
    if not torch.cuda.is_available():
        pytest.skip("This test requires GPU")

    layer = GeneralizedLinear(
        GeneralizedMatrix(512, 1536, blocksparse_layout="pixelfly(32)", use_triton=True,
                          lowrank_dim=32)
    ).cuda()

    with torch.no_grad():
        layer.bias[...] = torch.linspace(-10, 10, len(layer.bias)).cuda()
    input = torch.randn(3, 14, 512, requires_grad=True, device='cuda')

    out = layer(input)
    vec = torch.randn_like(out)
    (out * vec).sum().backward()

    grad_input = input.grad.clone()
    grad_params = [p.grad.clone() for p in layer.parameters()]
    input.grad = None
    layer.zero_grad(set_to_none=True)

    ref_out = layer.matrix(input) + layer.bias
    (ref_out * vec).sum().backward()

    assert torch.allclose(out, ref_out, rtol=0, atol=1e-6)
    assert torch.allclose(grad_input, input.grad, rtol=0, atol=1e-6)
    for (param_name, param), our_grad in zip(layer.named_parameters(), grad_params):
        assert torch.allclose(param.grad, our_grad, rtol=0, atol=1e-6), param_name
