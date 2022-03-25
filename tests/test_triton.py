import pytest
import torch

from lean_transformer import LeanTransformer, LeanTransformerConfig
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
        GeneralizedMatrix(512, 1536, blocksparse_layout="pixelfly(32)", blocksparse_backend='triton',
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

    assert torch.allclose(out, ref_out, rtol=0, atol=1e-5)
    assert torch.allclose(grad_input, input.grad, rtol=0, atol=1e-5)
    for (param_name, param), our_grad in zip(layer.named_parameters(), grad_params):
        assert torch.allclose(param.grad, our_grad, rtol=0, atol=1e-5), param_name


@pytest.mark.forked
@pytest.mark.parameterize("autocast, atol", [(False, 1e-6), (True, 0.05)])
def test_triton_ffn_transformer(autocast: bool = False, atol: float = 1e-5):
    if not torch.cuda.is_available():
        pytest.skip("This test requires GPU")

    model = LeanTransformer(LeanTransformerConfig(
        hidden_size=1024, num_hidden_layers=8, lowrank_dim=0,
        weight_layout="hypercube(32, folded=True)", blocksparse_backend='triton',
    ))

    model = model.cuda()
    input_embs = torch.randn(2, 8, 1024, device='cuda', requires_grad=True)
    mask = (torch.tensor([[1] * 8, [1] * 5 + [0] * 3], device='cuda')[:, None, None, :] - 1) * 1e6
    z = torch.randn(1024, device='cuda')

    model.zero_grad()
    model.set_optimizations(ffn_custom_grad=False)
    with torch.cuda.amp.autocast(autocast):
        hidden = model(input_embs, mask).last_hidden_state
        (hidden @ z).sum().backward()

        hidden_ref = hidden.clone()
        grads_ref = [p.grad.clone() for p in model.parameters()]
        grad_input_embs_ref = input_embs.grad.clone()

        input_embs.grad.zero_()
        model.zero_grad()
        model.set_optimizations(ffn_custom_grad=True)
        hidden = model(input_embs, mask).last_hidden_state
        (hidden @ z).sum().backward()

    assert torch.allclose(hidden, hidden_ref, rtol=0, atol=atol)
    assert torch.allclose(grad_input_embs_ref, input_embs.grad, rtol=0, atol=atol)

    for (name, param), grad_ref in zip(model.named_parameters(), grads_ref):
        assert torch.allclose(grad_ref, param.grad, rtol=0, atol=atol), name