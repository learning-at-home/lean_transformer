"""Pure-pytorch backend for block-sparse matrix multiplication."""

import torch
import torch.nn.functional as F

from lean_transformer.utils import maybe_script


@maybe_script
def blocksparse_matmul(input: torch.Tensor, weight: torch.Tensor, forward_indices: torch.Tensor) -> torch.Tensor:
    """
    :param input: tensor [*batch_dims, in_features]
    :param weight: tensor [in_features, active_blocks_per_input, block_size]
    :param forward_indices: the first output of get_butterfly_indices(...)
    :returns: tensor [*batch_dims, out_features]
    """
    assert input.shape[-1] == weight.shape[0]
    in_features, active_blocks_per_input, block_size = weight.shape
    num_input_blocks = in_features // block_size
    batch_dims = input.shape[:-1]
    input = input.flatten(0, -2)

    input_permuted = input.t().view(input.shape[1] // block_size, block_size, input.shape[0])
    output_blocks = torch.matmul(weight.view(num_input_blocks, -1, block_size), input_permuted)
    # ^-- shape: [num_input_blocks, (active_blocks_per_input * block_size), flat_batch_dims]

    blocks_for_indexing = output_blocks.view(num_input_blocks * active_blocks_per_input, block_size * input.shape[0])
    # ^-- shape: [(num_input_blocks * active_blocks_per_input),  (block_size, flat_batch_dims)]

    aggregated_blocks = F.embedding_bag(forward_indices, blocks_for_indexing, mode="sum")
    # ^-- shape: [num_ouput_blocks, (block_size, flat_batch_dims)]

    outputs = aggregated_blocks.view(-1, input.shape[0]).t()
    # ^-- shape: [flat_batch_dims, (num_output_blocks * block_size)] aka [flat_batch_dims, out_features]
    return outputs.view(batch_dims + outputs.shape[-1:])


@maybe_script
def blocksparse_matmul_backward(
        grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, backward_indices: torch.Tensor,
        input_requires_grad: bool = True, weight_requires_grad: bool = True):
    """Compute gradients of butterfly_matmul w.r.t. input and/or weight without relying on pytorch autograd"""
    assert input_requires_grad or weight_requires_grad, "computing backward but none of the inputs requires grad"
    grad_input = grad_weight = torch.empty(0)
    out_features = grad_output.shape[-1]
    in_features, active_blocks_per_input, block_size = weight.shape
    num_input_blocks = input.shape[-1] // block_size
    num_output_blocks = out_features // block_size
    grad_output_flat = grad_output.flatten(0, -2)
    input_flat = input.flatten(0, -2)

    flat_batch_dims = grad_output_flat.shape[0]

    grad_aggregated_blocks = grad_output_flat.t().reshape(num_output_blocks, (block_size * flat_batch_dims))
    # [num_output_blocks, (block_size, flat_batch_dims)]

    grad_blocks_for_indexing = F.embedding(backward_indices, grad_aggregated_blocks).flatten(0, -2)
    # ^-- shape: [(num_input_blocks * active_blocks_per_input),  (block_size, flat_batch_dims)]

    grad_output_blocks = grad_blocks_for_indexing.view(
        num_input_blocks, active_blocks_per_input * block_size, flat_batch_dims
    )
    # ^-- shape: [num_input_blocks, (active_blocks_per_input * block_size), flat_batch_dims]

    if input_requires_grad:
        grad_input_permuted = torch.matmul(
            weight.view(num_input_blocks, -1, block_size).permute(0, 2, 1), grad_output_blocks
        )
        grad_input = grad_input_permuted.flatten(0, -2).t().view(grad_output.shape[:-1] + input.shape[-1:])

    if weight_requires_grad:
        grad_weight = torch.matmul(
            grad_output_blocks, input_flat.t().view(num_input_blocks, block_size, flat_batch_dims).permute(0, 2, 1)
        ).view_as(weight)

    return grad_input, grad_weight
