import ast
import functools
import math
from typing import Optional, Tuple

import einops
import torch

from lean_transformer.utils import get_logger

from igraph import Graph

logger = get_logger(__name__)
REGISTERED_LAYOUTS = dict()


def register_blocksparse_layout(name: str):
    def _register(f: callable):
        if name in REGISTERED_LAYOUTS:
            logger.warning(f"Overwriting layout {name} with {f}.")
        REGISTERED_LAYOUTS[name] = f
        return f
    return _register


def get_blocksparse_layout(out_features: int, in_features: int, layout: str):
    """
    Get a boolean incidence matrix from matrix dimensions, block size and the layout descriptor.

    :param out_features: the first matrix dimension, corresponding to the number of output features in a linear layer
    :param in_features: the second matrix dimension, corresponding to the number of input features in a linear layer
    :param layout: a string that contains layout name and properties, similar to a function call, e.g. pixelfly()
      or pixelfly(block_size=128, stretch=False). Additional arguments are forwarded into the layout function.

    For standard layouts, search the code for the keyword "@register_blocksparse_layout".
    One can also implement a custom layout as follows:

    Example:
        >>> from lean_transformer import register_blocksparse_layout
        >>> from lean_transformer.models.gpt import LeanGPTConfig, LeanGPTModel
        >>> @register_blocksparse_layout("my_layout_name")
        >>> def make_my_layout(out_features: int, in_features: int, block_size: int, my_arg: str) -> torch.BoolTensor:
        >>>     # the first two arguments must be out_features and in_features the rest can be customized
        >>>     return DO_SOMETHING(...)
        >>> model = LeanGPTModel(LeanGPTConfig(..., blocksparse_layout="my_layout_name(block_size=64, my_arg='foo')"))

    """
    assert '(' in layout and layout.endswith(')'), "descriptor format must be name(foo=123, bar='456')"
    parsed = ast.parse(layout).body[0].value
    name = parsed.func.id
    args = [ast.literal_eval(arg) for arg in parsed.args]
    kwargs = {arg.arg: ast.literal_eval(arg.value) for arg in parsed.keywords}
    if name not in REGISTERED_LAYOUTS:
        raise ValueError(f"Unknown layout name {name}, supported layouts: {tuple(REGISTERED_LAYOUTS.keys())}")
    return REGISTERED_LAYOUTS[name](out_features, in_features, *args, **kwargs)


@register_blocksparse_layout("pixelfly")
def get_butterfly_layout(
    out_features: int,
    in_features: int,
    block_size: int = 256,
    butterfly_size: Optional[int] = None,
    n_factors: Optional[int] = None,
    stretch: bool = False,
) -> torch.BoolTensor:
    """

    A sum of blocky butterfly factors, as described in the paper: https://arxiv.org/abs/2112.00029 ;

    :note: pixelfly layout does NOT include the low-rank from the paper; use adapter_dim or lowrank_dim to add that.

    :param block_size: construct layout out of dense [block_size x block_size] weights
    :param butterfly_size: the logical size of butterfly blocks. Defaults to matrix size // blocks size.
      Providing a smaller value will result in butterfly matrix being tiled.
    :param n_factors: number of butterfly factors, equal to the number of edges minus one, the extra edge being a loop
    :param stretch: by default, non-square matrices will have stretched butterfly patterns,
      otherwise the square pattern will be repeated a given number of times
    """
    if butterfly_size is None:
        butterfly_size = 2 ** int(math.ceil(math.log2(min(in_features, out_features) / block_size)))
    assert out_features % in_features == 0 or in_features % out_features == 0, \
        "if matrix is not square, the longer dimension must be a multiple of the shorter dimension"
    assert out_features % block_size == 0 and in_features % block_size == 0
    log_n = int(math.log2(butterfly_size))
    n_factors = log_n if n_factors is None else n_factors
    if butterfly_size != 2 ** log_n or butterfly_size < 2:
        raise NotImplementedError("butterfly_size must be a power of 2")
    if not (1 <= n_factors <= log_n):
        raise NotImplementedError("n_factors must be a between 1 and log_2(butterfly_size)")

    twiddle = torch.ones(butterfly_size // 2, 2, 2)
    layout = sum(butterfly_factor_to_matrix(twiddle, index) for index in range(n_factors)).bool().int()
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


@register_blocksparse_layout("hypercube")
def get_hypercube_layout(
        out_features: int, in_features: int, block_size: int, stretch: bool = False, folded: bool = False):
    """
    An extension of pixelfly layout, based on its relation to hypercube (see https://tinyurl.com/hypercube-pixelfly)
    :param folded: add an extra edge connecting the farthest nodes (see https://en.wikipedia.org/wiki/Folded_cube_graph)
      This slightly decreases compute but significantly reduces the graph diameter.
    """
    smaller_features = min(out_features, in_features)
    assert math.log2(smaller_features / block_size).is_integer(), """layout dimension must be a power of 2"""
    assert out_features % smaller_features == 0 and in_features % smaller_features == 0
    cube_dimension = int(math.ceil(math.log2(min(in_features, out_features) / block_size)))
    layout = get_butterfly_layout(
        smaller_features, smaller_features, block_size, n_factors=cube_dimension, stretch=False
    )
    assert layout.shape[0] == layout.shape[1] == 2 ** cube_dimension
    if folded:
        for block_index in range(2 ** cube_dimension):
            block_binary = bin(block_index)
            assert block_binary.startswith('0b') and all(b in ('0', '1') for b in block_binary[2:])
            block_binary = '0b' + '0' * (cube_dimension + 2 - len(block_binary)) + block_binary[2:]
            opposite_binary = "0b" + ''.join('1' if b == '0' else '0' for b in block_binary[2:])
            opposite_index = int(opposite_binary[2:], base=2)
            assert block_index + opposite_index + 1 == 2 ** cube_dimension
            assert not layout[block_index, opposite_index].item()
            layout[block_index, opposite_index] = True

    if stretch:
        layout = layout[:, None, :, None].repeat(
            1, out_features // smaller_features, 1, in_features // smaller_features
        ).flatten(-2, -1).flatten(0, 1)
    else:
        layout = layout.repeat(out_features // smaller_features, in_features // smaller_features)
    return layout


@register_blocksparse_layout("kautz")
def get_kautz_layout(
        out_features: int, in_features: int, block_size: int, m: int, n: int, diagonal = True, stretch: bool = False):
    """
    A layout that uses Kautz graph (see https://en.wikipedia.org/wiki/Kautz_graph)
    :param diagonal: add an extra edge connecting each node with itself.
    """
    smaller_features = min(out_features, in_features)

    assert out_features % smaller_features == 0 and in_features % smaller_features == 0
    graph = Graph.Kautz(m, n)
    layout = torch.tensor(list(graph.get_adjacency()))
    assert smaller_features == layout.shape[0]*block_size
    if diagonal:
        layout += torch.eye(layout.shape[0], dtype=torch.bool)
    if stretch:
        layout = layout[:, None, :, None].repeat(
            1, out_features // smaller_features, 1, in_features // smaller_features
        ).flatten(-2, -1).flatten(0, 1)
    else:
        layout = layout.repeat(out_features // smaller_features, in_features // smaller_features)
    return layout


@register_blocksparse_layout("de_bruijn")
def get_de_bruijn_layout(
        out_features: int, in_features: int, block_size: int, m: int, n: int, stretch: bool = False):
    """
    A layout that uses De Bruijn graph (see https://en.wikipedia.org/wiki/De_Bruijn_graph)    """
    smaller_features = min(out_features, in_features)

    assert out_features % smaller_features == 0 and in_features % smaller_features == 0
    graph = Graph.De_Bruijn(m, n)
    layout = torch.tensor(list(graph.get_adjacency()))
    assert smaller_features == layout.shape[0]*block_size
    if stretch:
        layout = layout[:, None, :, None].repeat(
            1, out_features // smaller_features, 1, in_features // smaller_features
        ).flatten(-2, -1).flatten(0, 1)
    else:
        layout = layout.repeat(out_features // smaller_features, in_features // smaller_features)
    return layout


@register_blocksparse_layout("exponential")
def get_exponential_layout(
        out_features: int, in_features: int, block_size: int, diagonal = True, stretch: bool = False):
    """
    :param diagonal: add an extra edge connecting each node with itself.
    """
    smaller_features = min(out_features, in_features)

    assert out_features % smaller_features == 0 and in_features % smaller_features == 0
    layout = torch.zeros(smaller_features//block_size, smaller_features//block_size, dtype=torch.bool)
    for i in range(len(layout)):
        j = 1
        while j < len(layout):
            layout[i][(i+j)%len(layout)] = True
            j *= 2
    if diagonal:
        layout += torch.eye(layout.shape[0], dtype=torch.bool)
    if stretch:
        layout = layout[:, None, :, None].repeat(
            1, out_features // smaller_features, 1, in_features // smaller_features
        ).flatten(-2, -1).flatten(0, 1)
    else:
        layout = layout.repeat(out_features // smaller_features, in_features // smaller_features)
    return layout

def butterfly_factor_to_matrix(twiddle: torch.Tensor, factor_index: int) -> torch.Tensor:
    """
    Let b be the base (most commonly 2).
    Parameters:
        twiddle: (n // b, b, b)
        factor_index: an int from 0 to log_b(n) - 1
    """
    n_div_b, b, _ = twiddle.shape
    n = b * n_div_b
    log_b_n = int(math.log(n) / math.log(b))
    assert n == b ** log_b_n, f"n must be a power of {b}"
    assert twiddle.shape == (n // b, b, b)
    assert 0 <= factor_index <= log_b_n
    stride = b ** factor_index
    x = einops.rearrange(torch.eye(n), "bs (diagblk j stride) -> bs diagblk j stride", stride=stride, j=b)
    t = einops.rearrange(twiddle, "(diagblk stride) i j -> diagblk stride i j", stride=stride)
    out = torch.einsum("d s i j, b d j s -> b d i s", t, x)
    out = einops.rearrange(out, "b diagblk i stride -> b (diagblk i stride)")
    return out.t()  # Transpose because we assume the 1st dimension of x is the batch dimension


def get_indices_from_layout(layout: torch.BoolTensor) -> Tuple[torch.IntTensor, torch.LongTensor]:
    """
    Convert boolean incidence matrix to indices for F.embedding_bag

    :param layout: a boolean matrix [num output blocks, num input blocks]
    :returns: tuple (forward_indices, backward_indices), where
     - (forward) indices of non-zero blocks that contribute to each output -- assuming all input blocks are flattened
     - (backward) indices of output blocks to which a given input block contributes

    """
    num_output_blocks, num_input_blocks = layout.shape
    active_blocks_per_output = layout.sum(1).unique()
    assert len(active_blocks_per_output) == 1, "butterfly layout must have the same number of blocks per row"
    active_blocks_per_output = active_blocks_per_output.item()

    active_blocks_per_input = layout.sum(0).unique()
    assert len(active_blocks_per_input) == 1, "butterfly layout must have the same number of blocks per row"
    active_blocks_per_input = active_blocks_per_input.item()

    # which input blocks should be added for i-th output
    input_block_index = layout.nonzero()[:, 1].view(num_output_blocks, active_blocks_per_output)
    # which output blocks does j-th input contribute to
    output_block_index = layout.t().nonzero()[:, 1].view(num_input_blocks, active_blocks_per_input)

    # which of the active blocks from the corresponding input_block should be used for i-th output
    active_block_index = torch.where(
        torch.eq(
            output_block_index[input_block_index],
            torch.arange(len(input_block_index))[:, None, None],
        )
    )[-1].view(input_block_index.shape)

    forward_indices = input_block_index * active_blocks_per_input + active_block_index
    backward_indices = output_block_index
    return forward_indices.to(torch.int32), backward_indices.to(torch.int64)  # dtypes tuned for max throughput