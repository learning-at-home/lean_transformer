import pytest

import torch
import torch.nn as nn

import numpy as np

from lean_transformer.attn import LeanSelfAttention


@pytest.mark.forked
def test_lean_attn():
    torch.use_deterministic_algorithms(True)

    seq_length = 64
    num_seqs = 8
    hidden_dim = 128
    heads = 16

    gtruth_mha = nn.MultiheadAttention(hidden_dim, heads, bias=True,
                                       dropout=0, batch_first=True)

    for batch_step in [1, 2, 8, num_seqs * heads]:
        test_mha = LeanSelfAttention(hidden_dim, heads, dropout=0,
                                     pre_layer_norm=False, residual=False,
                                     checkpoint_attention_core=False,
                                     batched_attention_size=batch_step)

        test_mha.qkv_proj.weight = gtruth_mha.in_proj_weight
        test_mha.qkv_proj.bias = gtruth_mha.in_proj_bias
        test_mha.out_proj.weight = gtruth_mha.out_proj.weight
        test_mha.out_proj.bias = gtruth_mha.out_proj.bias

        device = torch.device('cpu')

        atol = 1e-6

        for _ in range(10):
            a = torch.randn((num_seqs, seq_length, hidden_dim), device=device)
            out0 = gtruth_mha(a, a, a)[0]
            out1 = test_mha(a)[0]
            out0.mean().backward()
            out1.mean().backward()
            out0 = out0.cpu().detach().numpy()
            out1 = out1.cpu().detach().numpy()
            assert np.allclose(out0, out1, atol=atol), f"{out0} {out1}"
