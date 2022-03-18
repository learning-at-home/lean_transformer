import math
from typing import Optional

import einops
import pytest
import torch
from lean_transformer.blocksparse import butterfly_factor_to_matrix, blocksparse_matmul, blocksparse_matmul_backward
from lean_transformer.blocksparse import get_blocksparse_layout, get_indices_from_layout


def get_butterfly_indices(out_features, in_features, *args, **kwargs):
    args_str = ', '.join(map(str, args))
    kwargs_str = ', '.join(f"{key}={value}" for key, value in kwargs.items())
    return get_indices_from_layout(get_blocksparse_layout(
        out_features, in_features, layout=f"pixelfly({args_str}, {kwargs_str})"))


@pytest.mark.forked
def test_block_sparse_matmul_internals(
    out_features=8192,
    in_features=4096,
    test_block: int = 2,
    block_size=256,
    butterfly_size=64,
    n_factors=6,
    stretch=True,
):
    """Test that butterfly matmul is indeed doing additive butterfly matrix multiplication"""

    butterfly_flat_indices, _backward_indices = get_butterfly_indices(
        out_features, in_features, block_size, butterfly_size, n_factors, stretch
    )
    assert butterfly_flat_indices.max() == butterfly_flat_indices.shape[1] * out_features // block_size - 1

    active_blocks_per_input = butterfly_flat_indices.numel() // (in_features // block_size)
    num_input_blocks = in_features // block_size
    weight = torch.randn(num_input_blocks * block_size, active_blocks_per_input, block_size).div_(100)

    # SETUP TEST CONDITIONS
    input = torch.randn(3, in_features)
    input.fill_(0)
    weight.fill_(1)
    input[:, test_block * block_size : test_block * (block_size + 1)].fill_(1)
    # END SETUP TEST CONDITIONS

    outputs = blocksparse_matmul(input, weight, butterfly_flat_indices)

    # BEGIN TEST CASE
    layout = reference_butterfly_layout_for_testing(
        out_features, in_features, block_size, butterfly_size, n_factors, stretch
    )
    for i in range(outputs.shape[1]):
        assert (outputs[:, i] != 0).all() == layout[i // block_size, test_block].item()
    # END TEST CASE


def reference_butterfly_layout_for_testing(
    out_features: int,
    in_features: int,
    block_size: int = 256,
    butterfly_size: int = 64,
    n_factors: Optional[int] = None,
    stretch: bool = False,
) -> torch.IntTensor:
    """
    Get a matrix [num_output_blocks, num_active_input_blocks] with int32 indices for additive butterfly
    Based on the original implementation from https://arxiv.org/abs/2112.00029 .

    :param stretch: by default, non-square matrices will have stretched butterfly patterns,
      otherwise the square pattern will be repeated a given number of times
    """
    assert (
        out_features % in_features == 0 or in_features % out_features == 0
    ), "matrix larger dimension must be divisible by the smaller one"
    assert out_features % block_size == 0 and in_features % block_size == 0
    log_n = int(math.log2(butterfly_size))
    n_factors = log_n if n_factors is None else n_factors
    if butterfly_size != 2 ** log_n or butterfly_size < 2:
        raise NotImplementedError("butterfly_size must be a power of 2")
    if not (1 <= n_factors <= log_n):
        raise NotImplementedError("n_factors must be a between 1 and log_2(butterfly_size)")

    twiddle = torch.ones(butterfly_size // 2, 2, 2)
    layout = sum(butterfly_factor_to_matrix(twiddle, index) for index in range(n_factors))
    layout = layout.bool().int()
    # Convert from (butterfly_size, butterfly_size) mask to (out_features, in_features) mask
    layout = einops.repeat(
        layout,
        "b b1 -> (b f) (b1 f1)",
        f=out_features // butterfly_size,
        f1=in_features // butterfly_size,
    )
    # Convert from (out_features, in_features) mask to
    # (out_features // block_size, in_features // block_size) mask
    layout = einops.rearrange(
        layout,
        "(p blksz) (r blksz1) -> p r (blksz blksz1)",
        blksz=block_size,
        blksz1=block_size,
    )

    layout = (layout > 0).any(dim=-1)  # [out_features // block_size, in_features // block_size]
    if not stretch:
        out_blocks, in_blocks = layout.shape
        if out_blocks > in_blocks:
            ratio = out_blocks // in_blocks
            layout = layout.view(out_blocks // ratio, ratio, in_blocks).permute(1, 0, 2).reshape_as(layout)
        elif out_blocks < in_blocks:
            ratio = in_blocks // out_blocks
            layout = layout.view(out_blocks, in_blocks // ratio, ratio).permute(0, 2, 1).reshape_as(layout)
    return layout


@pytest.mark.forked
def test_butterfly_gradients():
    out_features = 3072 * 4
    in_features = 3072
    block_size = 96
    n_factors = None
    butterfly_size = None
    stretch = False

    torch.manual_seed(42)
    forward_indices, backward_indices = get_butterfly_indices(
        out_features, in_features, block_size, butterfly_size, n_factors, stretch
    )

    input = torch.randn(3, 2, in_features, requires_grad=True)
    weight = torch.randn(
        in_features, forward_indices.shape[1] * out_features // in_features, block_size, requires_grad=True
    )
    grad_output = torch.randn(3, 2, out_features)

    with torch.no_grad():
        our_grad_input, our_grad_weight = blocksparse_matmul_backward(grad_output, input, weight, backward_indices)

    with torch.enable_grad():
        out = blocksparse_matmul(input, weight, forward_indices)

    out.backward(grad_output)
    assert torch.allclose(our_grad_input, input.grad)
    assert torch.allclose(our_grad_weight, weight.grad)
