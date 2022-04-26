import math
from functools import lru_cache
from typing import Optional

from torch import nn as nn
from transformers import PretrainedConfig

from lean_transformer.attn import SimpleAttentionCore, RotaryAttentionCore, RotaryEmbeddings
from lean_transformer.blocksparse import GeneralizedLinear, GeneralizedMatrix
from lean_transformer.utils import ACT2FN


class LeanTransformerConfig(PretrainedConfig):
    r"""

    :param hidden_size: main hidden dimension of a transformer, used as inputs and outputs of all layers
    :param intermediate_size: a (typically larger) hidden dimension where activation is applied
    :param num_attention_heads: number of heads in each attention layer, as defined in the original transformer

    :param num_hidden_layers: the total number of layers before sharing
    :param reversible: if True, use reversible layer order as defined ReFormer ( arXiv:2001.04451 ). This dramatically
      reduces memory usage, but slightly increases computation for the backward pass (same as in gradient checkpoints)

    :param num_hidden_groups: number of ALBERT-like layer groups with independent parameters
    :param num_inner_groups: by default, each layer group contains one attention and one FFN layer. Setting this to
      more than 1 will result in multiple (attn, ffn) pairs stacked on top of each other in each of num_hidden_groups

    :param share_large_matrices: False / True or an integer. False means all ffn and attention layers are independent.
      if True, layers reuse a set of shared matrices (e.g. one for all QKV attentions, another for all FFN projections)
      if an integer, use this number of sets of shared matrices (consecutive, each is num_hidden_layers // num_matrices)
    :param num_inner_matrices: if sharing is used, this enables using several interleaved shared matrices per set

    :param adapter_dim: if share_large_matrices is used, each layer can make LoRA-like adapters to the shared matrices.
      The adapter_dim corresponds to a hidden dimension of that adapter (see arXiv:2106.09685 for LoRA)
    :param weight_layout: if specified, replaces weight matrices in FFN and attention with block-sparse matrices,
      defined by this layout. For instance, "pixelfly(block_size=128)" is Pixelated Buttefly (arXiv:2112.00029 ).
      This does not affect embeddings or attention logits.
    :param lowrank_dim: if specified, add a (shared) low-rank component to the block-sparse matrix, as recommended
      in the PixelFly paper ( arXiv:2112.00029 ). The difference from adapter_dim is that adapters are NOT shared.
    :param hidden_act: activation function for FFN layers, either string or callable
    :param gated: use gated activations based on https://arxiv.org/abs/2002.05202 and https://arxiv.org/abs/2102.11972
      note: gated activations require 1.5x more parameters compared to their non-gated variants.
    :param attn_qkv_bias: whether or not to use biases in attention qkv projection
    :param out_proj_bias: whether or not to use biases in output projections of both attention and ffn,
      defaults to True unless post_layer_norm is enabled -- since post-norm already has a bias component
    :param post_layer_norm: if set, applies an additional layer norm to projected attention outputs before residuals,
       as proposed in the CogView paper ( arXiv:2105.13290 ). This is meant to make fp16 training
       more stable for deep transformers. This technique is also a part of NormFormer ( arXiv:2110.09456 )

    :param hidden_dropout_prob: dropout applied to the outputs of each attention and FFN layer right before residual;
    :param attention_probs_dropout_prob: if specified, randomly prevent attention head from drop looking at some tokens;
    :note: Lan et al. ( arXiv:1909.11942 ) *disable* Dropout for pre-training, then re-enable it for fine-tuning

    :param layer_norm_eps: see layer_norm_eps in torch.nn.functional.layer_norm
    :param attention_type: either "simple" (as in BERT) or "rotary" (arXiv:2104.09864 , used in GPT-J-6B)
    :param rotary_embedding_base: base for computing the rotation periods, only if attention_type is "rotary"

    :param initializer_range: standard deviation for gaussian noise used when initializing weight matrices, defaults
     to SmallInit (see https://arxiv.org/pdf/1910.05895.pdf section 2.2) = sqrt(2 / (5 * hidden_size))
    :param initializer_adjust_sparse: if True, scale initializer range for sparse matrices by 1/sqrt(density)
    :note: the initialized range is **not** applied by default, it requires calling model.apply(model.init_weights)!

    :param kwargs: additional keyword arguments used by base PretrainedModel in huggingface transformers

    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_hidden_groups: Optional[int] = None,
        num_inner_groups: int = 1,
        share_large_matrices: int = 0,
        num_inner_matrices: int = 1,
        adapter_dim: int = 0,
        num_attention_heads: int = 64,
        intermediate_size: Optional[int] = None,
        weight_layout: Optional[str] = None,
        blocksparse_backend: str = 'native',
        lowrank_dim: int = 0,
        hidden_act: str = "gelu_fused",
        hidden_act_gated: bool = False,
        attn_qkv_bias: bool = True,
        out_proj_bias: Optional[bool] = None,
        post_layer_norm: bool = False,
        reversible: bool = False,
        hidden_dropout_prob: float = 0,
        attention_probs_dropout_prob: float = 0,
        attention_type: str = "simple",
        layer_norm_eps: float = 1e-12,
        rotary_embedding_base: int = 10_000,
        initializer_range: Optional[float] = None,
        initializer_adjust_sparse: bool = True,
        **kwargs,
    ):
        if "sandwich_norm" in kwargs:
            raise ValueError("sandwich_norm was renamed, please use pre_layer_norm=True and post_layer_norm=True")
        if "block_size" in kwargs:
            raise ValueError("block_size was renamed, use weight_layout='pixelfly(block_size=128)'")
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.adapter_dim = adapter_dim
        self.lowrank_dim = lowrank_dim
        self.weight_layout = weight_layout
        self.blocksparse_backend = blocksparse_backend

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups if num_hidden_groups is not None else self.num_hidden_layers
        self.num_inner_groups = num_inner_groups
        self.total_num_layer_groups = self.num_hidden_groups * self.num_inner_groups

        assert isinstance(share_large_matrices, (bool, int)) and share_large_matrices >= 0
        assert num_inner_matrices <= 1 or share_large_matrices, \
            "inner_shared_matrices is only used if share_large_matrices >= 1"
        self.share_large_matrices = bool(share_large_matrices)
        self.num_shared_matrices = int(share_large_matrices) if share_large_matrices else self.total_num_layer_groups
        self.num_inner_matrices = num_inner_matrices
        self.total_shared_matrix_sets = self.num_shared_matrices * self.num_inner_matrices
        assert self.total_shared_matrix_sets <= self.total_num_layer_groups, \
            f"there are {self.total_shared_matrix_sets} but only {self.total_num_layer_groups} layers to share among"

        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_act_gated = hidden_act_gated
        self.layer_norm_eps = layer_norm_eps
        self.attn_qkv_bias = attn_qkv_bias
        self.out_proj_bias = out_proj_bias if out_proj_bias is not None else not post_layer_norm
        self.post_layer_norm = post_layer_norm
        self.reversible = reversible

        self.attention_type = attention_type
        self.rotary_embedding_base = rotary_embedding_base

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        if initializer_range is None:
            initializer_range = math.sqrt(2 / (5 * self.hidden_size))
            # note: this default values is based on SmallInit (see https://arxiv.org/pdf/1910.05895.pdf section 2.2)
        self.initializer_range = initializer_range
        self.initializer_adjust_sparse = initializer_adjust_sparse

    def __hash__(self):
        return hash("\t".join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")))

    @lru_cache()
    def _get_rotary_cache(self):
        assert self.attention_type == "rotary"
        return RotaryEmbeddings(self.hidden_size // self.num_attention_heads, self.rotary_embedding_base)

    def get_attention_core(self):
        if self.attention_type == "simple":
            return SimpleAttentionCore(
                self.hidden_size, self.num_attention_heads, attention_probs_dropout=self.attention_probs_dropout_prob
            )
        elif self.attention_type == "rotary":
            return RotaryAttentionCore(
                self.hidden_size,
                self.num_attention_heads,
                self._get_rotary_cache(),
                attention_probs_dropout=self.attention_probs_dropout_prob,
            )
        else:
            raise NotImplementedError(f"Unsupported attention type: {self.attention_type}")

    @lru_cache()
    def get_activation_callable(self):
        hidden_act_callable = ACT2FN[self.hidden_act] if not callable(self.hidden_act) else self.hidden_act
        assert callable(hidden_act_callable)
        return hidden_act_callable

    def get_linear_layer(self, key: str, index: int, in_features: int, out_features: int, bias: bool) -> nn.Linear:
        if not self.share_large_matrices and self.adapter_dim != 0:
            raise ValueError("not sharing matrices => adapter_dim should be 0. Use lowrank_dim instead.")

        assert self.num_hidden_layers == self.total_num_layer_groups, "sharing is not implemented because yozh was lazy"
        assert self.lowrank_dim != 0, "need lowrank"
        return VoidLinear(in_features, out_features, lowrank_dim=self.lowrank_dim, bias=bias)

    @lru_cache(maxsize=None)
    def get_weight_matrix(self, key: str, index: int) -> Optional[GeneralizedMatrix]:
        """
        Create a weight matrix for use in transformer layers, optionally use block-wise sparsity

        :param key: a string identifier of matrix within a transformer layer, e.g. "self_attn_qkv"
        :param index: an index of a shared matrix set, if there is more than one
        :note: even if index is not used in this function, it is necessary to ensure that lru_cache works correctly
        """
        raise NotImplementedError()
        assert 0 <= index <= self.total_shared_matrix_sets
        if key == "self_attn_qkv":
            return GeneralizedMatrix(self.hidden_size, self.hidden_size * 3, self.weight_layout, self.lowrank_dim,
                                     blocksparse_backend=self.blocksparse_backend)
        if key == "self_attn_out":
            return GeneralizedMatrix(self.hidden_size, self.hidden_size, self.weight_layout, self.lowrank_dim,
                                     blocksparse_backend=self.blocksparse_backend)
        if key == "ffn_first":
            ffn_hidden_including_gate = self.intermediate_size * (2 if self.hidden_act_gated else 1)
            return GeneralizedMatrix(self.hidden_size, ffn_hidden_including_gate, self.weight_layout, self.lowrank_dim,
                                     blocksparse_backend=self.blocksparse_backend)
        if key == "ffn_second":
            return GeneralizedMatrix(self.intermediate_size, self.hidden_size, self.weight_layout, self.lowrank_dim,
                                     blocksparse_backend=self.blocksparse_backend)

        raise NotImplementedError(f"Unexpected matrix key: {key}")

    def init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, GeneralizedMatrix):
            module.initialize_(self.initializer_range, adjust_for_sparsity=self.initializer_adjust_sparse)
        elif isinstance(module, GeneralizedLinear):
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math


class VoidLinear(nn.Module):
    def __init__(
            self, in_features: int, out_features: int, *, lowrank_dim: int = 128,
            block_size: int = 64, codebook_size: int = 256, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features, self.lowrank_dim = in_features, out_features, lowrank_dim
        assert out_features % block_size == 0
        assert in_features % block_size == 0
        self.out_blocks, self.in_blocks = out_features // block_size, in_features // block_size

        blocky_shape = (self.out_blocks, self.in_blocks, block_size, block_size)
        self.register_buffer('zm', torch.randint(0, codebook_size, blocky_shape, dtype=torch.uint8), persistent=True)
        self.codebooks = nn.Parameter(torch.empty(self.out_blocks * self.in_blocks, codebook_size))

        self.lowrank_first = nn.Parameter(torch.empty(lowrank_dim, in_features))
        self.lowrank_second = nn.Parameter(torch.empty(out_features * 2, lowrank_dim))
        self.grid_scale = nn.Parameter(torch.ones(out_features // 8, 1, in_features // 8, 1))
        self.grid_bias = nn.Parameter(torch.zeros(out_features // 8, 1, in_features // 8, 1))
        self.scale = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        print('CHECKSUM:', self.zm.sum())
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                print("BARRIER", flush=True)
        nn.init.normal_(self.codebooks, mean=0.0, std=math.sqrt(2.0 / (5 * min(out_features, in_features))))
        nn.init.xavier_uniform_(self.lowrank_first)
        nn.init.zeros_(self.lowrank_second)
        nn.init.ones_(self.scale)
        nn.init.ones_(self.grid_scale)
        nn.init.zeros_(self.grid_bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        dtype = torch.float16 if torch.is_autocast_enabled() else input.dtype
        random_matrix = cast_and_gather(self.zm, self.codebooks.to(dtype))
        random_matrix = torch.addcmul(
            self.grid_bias.to(dtype),
            random_matrix.view(self.grid_scale.shape[0], 8, self.grid_scale.shape[2], 8),
            self.grid_scale.to(dtype)
        ).view(self.out_features, self.in_features)
        if torch.is_autocast_enabled():
            baseline = F.linear(input, random_matrix)
        else:
            baseline = F.linear(input.to(random_matrix.dtype), random_matrix).to(input.dtype)
        hid = F.linear(input, self.lowrank_first)
        bias_or_zeros = torch.zeros_like(self.scale) if self.bias is None else self.bias
        intercept = torch.cat([self.scale, bias_or_zeros], dim=0)
        multiplicative, additive = F.linear(hid, self.lowrank_second, intercept).split(self.out_features, dim=-1)
        return torch.addcmul(additive, multiplicative, baseline)


def cast_and_gather(zm: torch.ByteTensor, codebooks: nn.Parameter) -> torch.Tensor:
    """
    A crutch that selects values from codebooks using int8 indices, bypassing the limitation that indices must be int64
    This function temporarily casts indices (zm) to 64-bit and selects values without saving them for backward.
    As such, only one set of int64 tensors will be materialized at a time.
    """
    with torch.cuda.amp.autocast(enabled=False):
        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(_cast_and_gather_inner, zm, codebooks, preserve_rng_state=False)
        else:
            return _cast_and_gather_inner(zm, codebooks)


def _cast_and_gather_inner(zm: torch.ByteTensor, codebooks: torch.Tensor) -> torch.Tensor:
    out_blocks, in_blocks, block_size, _ = zm.shape
    zm_flat_blocks = zm.view(out_blocks * in_blocks, block_size * block_size)
    flat_blocks_permuted = torch.gather(codebooks, 1, zm_flat_blocks.long()).view(zm.shape).swapaxes_(1, 2)
    # ^-- shape: [out_blocks, block_size, in_blocks, block_size]

    return flat_blocks_permuted.reshape(out_blocks * block_size, in_blocks * block_size)  # <-- this creates a copy!

