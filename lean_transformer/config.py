import math
from functools import lru_cache
from typing import Optional

from torch import nn as nn
from transformers import PretrainedConfig

from lean_transformer import RotaryEmbeddings, SimpleAttentionCore, RotaryAttentionCore, ACT2FN, GeneralizedLinear, \
    GeneralizedMatrix


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
    :param block_size: if specified, replaces weight matrices in FFN and attention with block-sparse butterfly matrices,
      as defined in the Pixelated Buttefly ( arXiv:2112.00029 ). This does not affect embeddings or attention logits.
    :param lowrank_dim: if specified, add a (shared) low-rank component to the block-sparse matrix, as recommended
      in the PixelFly paper ( arXiv:2112.00029 ). The difference from adapter_dim is that adapters are NOT shared.
    :param hidden_act: activation function for FFN layers, either string or callable
    :param gated: use gated activations based on https://arxiv.org/abs/2002.05202 and https://arxiv.org/abs/2102.11972
      note: gated activations require 1.5x more parameters compared to their non-gated variants.
    :param attn_qkv_bias: whether or not to use biases in attention qkv projection
    :param out_proj_bias: whether or not to use biases in output projections of both attention and ffn,
      defaults to True unless sandwich_norm is enabled -- since sandwich norm already has a bias component
    :param sandwich_norm: if set, applies an additional layer norm to projected attention outputs before residuals,
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
        block_size: int = 0,
        lowrank_dim: int = 0,
        hidden_act: str = "gelu_fused",
        hidden_act_gated: bool = False,
        attn_qkv_bias: bool = True,
        out_proj_bias: Optional[bool] = None,
        sandwich_norm: bool = False,
        reversible: bool = False,
        hidden_dropout_prob: float = 0,
        attention_probs_dropout_prob: float = 0,
        attention_type: str = "simple",
        layer_norm_eps: float = 1e-12,
        rotary_embedding_base: int = 10_000,
        initializer_range: Optional[float] = None,
        **kwargs,
    ):
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.adapter_dim = adapter_dim
        self.lowrank_dim = lowrank_dim
        self.block_size = block_size

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
        self.out_proj_bias = out_proj_bias if out_proj_bias is not None else not sandwich_norm
        self.sandwich_norm = sandwich_norm
        self.reversible = reversible

        self.attention_type = attention_type
        self.rotary_embedding_base = rotary_embedding_base

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        if initializer_range is None:
            initializer_range = math.sqrt(2 / (5 * self.hidden_size))
            # note: this default values is based on SmallInit (see https://arxiv.org/pdf/1910.05895.pdf section 2.2)
        self.initializer_range = initializer_range

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

        matrix_outer_index = (index * self.num_shared_matrices) // self.total_num_layer_groups
        matrix_inner_index = index % self.num_inner_matrices
        matrix_index = matrix_outer_index * self.num_inner_matrices + matrix_inner_index

        weight_matrix = self.get_weight_matrix(key, matrix_index)
        assert tuple(weight_matrix.shape) == (out_features, in_features)
        return GeneralizedLinear(weight_matrix, self.adapter_dim, bias)

    @lru_cache(maxsize=None)
    def get_weight_matrix(self, key: str, index: int) -> Optional[GeneralizedMatrix]:
        """
        Create a weight matrix for use in transformer layers, optionally use block-wise sparsity

        :param key: a string identifier of matrix within a transformer layer, e.g. "self_attn_qkv"
        :param index: an index of a shared matrix set, if there is more than one
        :note: even if index is not used in this function, it is necessary to ensure that lru_cache works correctly
        """
        assert 0 <= index <= self.total_shared_matrix_sets
        if key == "self_attn_qkv":
            return GeneralizedMatrix(self.hidden_size, self.hidden_size * 3, self.block_size, self.lowrank_dim)
        if key == "self_attn_out":
            return GeneralizedMatrix(self.hidden_size, self.hidden_size, self.block_size, self.lowrank_dim)
        if key == "ffn_first":
            return GeneralizedMatrix(self.hidden_size, self.intermediate_size * (2 if self.hidden_act_gated else 1),
                                     self.block_size, self.lowrank_dim)
        if key == "ffn_second":
            return GeneralizedMatrix(self.intermediate_size, self.hidden_size, self.block_size, self.lowrank_dim)

        raise NotImplementedError(f"Unexpected matrix key: {key}")

    def init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, GeneralizedMatrix):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
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