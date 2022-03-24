from typing import Tuple, Sequence

import torch
try:
    import triton.ops
    from triton.ops.blocksparse.matmul import matmul, _matmul
except ModuleNotFoundError as e:
    triton = matmul = e  # triton will not work

from lean_transformer.utils import pad_to_multiple


class TritonMatmulForLinearLayer(matmul):
    """A thin wrapper over triton block-sparse matmul that supports easier integration into custom autograd functions"""
    def __init__(self, layout: torch.BoolTensor, block_size: int):
        assert not isinstance(triton, Exception), f"triton is not available: {triton}"
        super().__init__(layout.cpu(), block_size, 'dds', trans_a=False, trans_b=True)

    def __call__(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        :param input: tensor [*batch_dims, in_features]
        :param weight: tensor [in_features, active_blocks_per_input, block_size]
        :returns: output tensor [*batch_dims, out_features]
        """
        input_flat = input.flatten(0, -2)
        input_padded = pad_to_multiple(input_flat, multiple=16, dims=0)
        output_flat = super()(input_padded[None, None, ...], weight).flatten(0, -2)
        output = output_flat[:input_flat.shape[0]].view(*input.shape[:-1], output_flat.shape[-1])
        return output

    def forward_functional(self, input: torch.Tensor, weight: torch.Tensor
                           ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """ :returns: output tensor, tensors_to_save """
        c_lut, c_num_locks, c_width, c_packs, _, _, _, _, _, _, _, _ = self.make_lut(input.dtype, input.device)
        input_padded = pad_to_multiple(input.flatten(0, -2), multiple=16, dims=0)
        input_padded, weight = self._validate_inputs(input_padded, weight)

        output = _matmul.fn[self.mode](
            input_padded, weight, self.trans_a, self.trans_b, False, self.spdims, self.block,
            c_lut, c_num_locks, c_width, c_packs
        )
        # remove padding and restore original shape
        output = output.flatten(0, -2)[:input_padded.shape[0]].view(*input.shape[:-1], output.shape[-1])
        return output, (input_padded, weight)

    def backward_functional(
            self, grad_output: torch.Tensor, saved_tensors: Sequence[torch.Tensor], needs_input_grad: Tuple[bool, ...]):
        grad_input = grad_weight = grad_output_padded = None
        input_padded, weight = saved_tensors
        _, _, _, _, dx_lut, dx_num_locks, dx_width, dx_packs, dw_lut, dw_num_locks, dw_width, dw_packs = \
            self.make_lut(input_padded.dtype, input_padded.device)
        if any(needs_input_grad):
            grad_output_padded = pad_to_multiple(grad_output.flatten(0, -2), multiple=16, dims=0)

        if needs_input_grad[0]:
            grad_input_mode = self.mode[1] + self.mode[0] + self.mode[2]
            grad_input_padded = _matmul.fn[grad_input_mode](
                grad_output_padded, weight, False, not self.trans_b, self.trans_a, self.spdims, self.block,
                dx_lut, dx_num_locks, dx_width, dx_packs
            )
            original_num_rows = grad_output.numel() // grad_output.shape[-1]
            grad_input = grad_input_padded.flatten(0, -2)[:original_num_rows]
            grad_input = grad_input.view(*grad_output.shape[:-1], input_padded.shape[-1])
            del grad_input_padded
        if needs_input_grad[1]:
            mode_db = self.mode[2] + self.mode[1] + self.mode[0]
            grad_weight = _matmul.fn[mode_db](
                input_padded, grad_output_padded, not self.trans_a, False, self.trans_b, self.spdims, self.block,
                dw_lut, dw_num_locks, dw_width, dw_packs
            )
        return grad_input, grad_weight
