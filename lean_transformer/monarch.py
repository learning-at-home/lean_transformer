import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
from functools import partial
from typing import Sequence


class MonarchLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 in_dims: Sequence[int], out_dims: Sequence[int],
                 bias: bool = True, checkpoint: bool = False,
                 ):
        """
        Monarch linear layer, a generalization of https://arxiv.org/abs/2204.00595

        Ths implementation interprets Monarch as a product over an M by M grid (in_features=M ^ 2).
        The first product applies over all rows of the grid, the second runs over columns.
        In general, the grid may have uneven size or more than 2 dimensions.

        In the 2d case, the two products use [M x M x M] weight tensors. In the general case,
        it uses grid_dim weight tensors of shape [grid_numel / in_dims[i], in_dims[i], out_dims[i]].

        :param in_features: input dimension, same as in nn.Linear
        :param out_features: output dimension, same as in nn.Linear
        :param in_dims: a tuple of numbers that multiply to in_features, see example below
        :param out_dims: a tuple of numbers that multiply to out_features, see example below
        :param bias: whether or not to use a bias term, same as in nn.Linear
        :param checkpoint: if True, apply gradient checkpointing over this entire layer.
           This adds ~30% compute overhead for forward+backward, but reduces the memory overhead;
           otherwise, monarch must to store ndim - 1 additional tensors for intermediate activations.

        :example:

        >>> # classic monarch:
        >>> MonarchLinear(in_features=1024, in_dims=(32, 32), out_features=1024, out_dims=(32, 32))
        >>> # generalization to rectangular matrices
        >>> MonarchLinear(in_features=1024, in_dims=(32, 32), out_features=4096, out_dims=(64, 64))
        >>> MonarchLinear(in_features=1024, in_dims=(32, 32), out_features=1536, out_dims=(32, 48))
        >>> # generalization to higher dimension
        >>> MonarchLinear(in_features=4096, in_dims=(16, 16, 16), out_features=4096, out_dims=(16, 16, 16))
        >>> MonarchLinear(in_features=4096, in_dims=(16, 16, 16), out_features=1536, out_dims=(8, 12, 16))

        """
        super().__init__()
        assert len(in_dims) == len(out_dims) and len(in_dims) > 1
        assert np.prod(in_dims) == in_features
        assert np.prod(out_dims) == out_features
        self.in_features, self.out_features = in_features, out_features
        self.in_dims, self.out_dims = in_dims, out_dims
        self.checkpoint = checkpoint

        # construct weight tensors by keeping track of intermediate tensor dimension at each step
        self.weights = nn.ParameterList()
        current_numel = np.prod(in_dims)
        assert current_numel == in_features
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.weights.append(nn.Parameter(torch.empty(current_numel // in_dim, in_dim, out_dim)))
            current_numel = current_numel // in_dim * out_dim
        assert current_numel == out_features
        self.register_parameter('bias', nn.Parameter(torch.empty(out_features)) if bias else None)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.0):
        # initialize, re-scale to account for the number of multiplied tensors
        init_std = (gain / np.sqrt(self.in_features)) ** (1 / len(self.in_dims))
        for weight in self.weights:
            nn.init.normal_(weight, std=init_std)
        if self.bias is not None:
            bound = 1 / np.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, _inside_checkpoint: bool = False):
        if self.checkpoint and not _inside_checkpoint and torch.is_grad_enabled():
            return checkpoint(partial(self.forward, _inside_checkpoint=True),
                              input if input.requires_grad else input.detach().requires_grad_(True),
                              preserve_rng_state=False)
        input_shape = input.shape
        tensor = input.view(-1, *self.in_dims)
        # shape: [flat_batch_size, in_dim[0], ..., in_dim[N]]

        del input
        tensor = tensor.permute(*np.roll(range(len(self.in_dims) + 1), -2))
        # new shape: [in_dim[1], ..., in_dim[N - 1], flat_batch_size, in_dim[0]]

        for i in range(len(self.weights)):
            # loop maintains tensor in the following shape: [*all_dims_except_i, batch, dim[i]]

            tensor = torch.bmm(
                tensor.flatten(0, -3), self.weights[i]
            ).view(*tensor.shape[:-1], -1)
            # ^-- BMM, output: [*other_dims, batch, out_dim[i]]
            #     left input:  [*other_dims, batch, in_dim[i]]
            #     right_input: [*other_dims, in_dim[i], out_dim[i]]

            # prepare next step, from [*other_dims, batch, out_dim[i]] to [*other_dims, batch, in_dim[i + 1]]
            tensor = tensor.swapaxes_(-1, i)
            # note: we can swap in-place because bmm does not need outputs for backprop

        # after loop: [out_dim[0], ..., out_dim[N - 1], batch]
        tensor = tensor.flatten(0, -2).swapaxes_(0, 1)
        tensor = tensor.reshape(*input_shape[:-1], -1)
        if self.bias is not None:
            tensor += self.bias
        return tensor
