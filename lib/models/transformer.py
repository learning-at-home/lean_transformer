import math
from functools import lru_cache
from typing import Optional, Tuple

from torch import nn as nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput

from lib.modules import LeanFFN, LeanSelfAttention
from lib.modules.attn import RotaryAttentionCore, RotaryEmbeddings, SimpleAttentionCore
from lib.modules.linear import SemiSharedLinear, SharedMatrix
from lib.modules.sequence import ActiveKwargs, ReversibleWithKwargs, SequentialWithKwargs


class LeanTransformerConfig(PretrainedConfig):
    r"""
    Similar to AlbertConfig, but its a lean transformer and we didn't write its description yet
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_hidden_groups: Optional[int] = None,
        num_inner_groups: int = 1,
        share_large_matrices: bool = False,
        adapter_dim: int = 0,
        num_attention_heads: int = 64,
        intermediate_size: int = 16384,
        block_size: int = 0,
        lowrank_dim: int = 0,
        hidden_act: str = "gelu_new",
        hidden_act_gated: bool = False,
        sandwich_norm: bool = False,
        reversible: bool = False,
        hidden_dropout_prob: float = 0,
        attention_probs_dropout_prob: float = 0,
        layer_norm_eps: float = 1e-12,
        position_embedding_type: str = "rotary",
        max_position_embeddings: int = 512,
        rotary_embedding_base: int = 10_000,
        initializer_range: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.share_large_matrices = share_large_matrices
        self.adapter_dim = adapter_dim
        self.lowrank_dim = lowrank_dim
        self.block_size = block_size

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups if num_hidden_groups is not None else self.num_hidden_layers
        self.num_inner_groups = num_inner_groups

        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act_gated = hidden_act_gated
        self.sandwich_norm = sandwich_norm
        self.reversible = reversible

        if position_embedding_type == "absolute":
            assert max_position_embeddings is not None
        self.position_embedding_type = position_embedding_type
        self.rotary_embedding_base = rotary_embedding_base
        self.max_position_embeddings = max_position_embeddings

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
        assert self.position_embedding_type == "rotary"
        return RotaryEmbeddings(self.hidden_size // self.num_attention_heads, self.rotary_embedding_base)

    def get_attention_core(self):
        if self.position_embedding_type == "absolute":
            return SimpleAttentionCore(
                self.hidden_size, self.num_attention_heads, attention_probs_dropout=self.attention_probs_dropout_prob
            )
        elif self.position_embedding_type == "rotary":
            return RotaryAttentionCore(
                self.hidden_size,
                self.num_attention_heads,
                self._get_rotary_cache(),
                attention_probs_dropout=self.attention_probs_dropout_prob,
            )
        else:
            raise NotImplementedError(f"Unsupported embedding type: {self.position_embedding_type}")

    def get_input_position_embeddings(self) -> Optional[nn.Embedding]:
        if self.position_embedding_type == "absolute":
            return nn.Embedding(self.max_position_embeddings, self.embedding_size)
        elif self.position_embedding_type == "rotary":
            return None
        else:
            raise NotImplementedError(f"Unsupported embedding type: {self.position_embedding_type}")

    def get_token_type_embeddings(self) -> Optional[nn.Embedding]:
        return nn.Embedding(self.type_vocab_size, self.embedding_size) if self.type_vocab_size else None

    def get_linear_layer(self, key: str, in_features: int, out_features: int, bias: bool):
        assert self.adapter_dim == 0 or self.share_large_matrices, "not sharing matrices => adapter_dim should be 0"
        if not self.share_large_matrices and self.block_size == 0:
            return nn.Linear(in_features, out_features, bias)

        elif self.share_large_matrices:
            shared_matrix = self.get_shared_matrix(key)
            assert tuple(shared_matrix.shape) == (out_features, in_features)
            return SemiSharedLinear(shared_matrix, self.adapter_dim, bias)

        else:
            assert not self.share_large_matrices and self.block_size > 0 and self.adapter_dim == 0
            shared_matrix = SharedMatrix(in_features, out_features, self.block_size, self.lowrank_dim)
            return SemiSharedLinear(shared_matrix, self.adapter_dim, bias)  # not actually shared

    @lru_cache()
    def get_shared_matrix(self, key: str) -> Optional[SharedMatrix]:
        assert self.share_large_matrices
        if key == "self_attn_qkv":
            return SharedMatrix(self.hidden_size, self.hidden_size * 3, self.block_size, self.lowrank_dim)
        if key == "self_attn_out":
            return SharedMatrix(self.hidden_size, self.hidden_size, self.block_size, self.lowrank_dim)
        if key == "ffn_first":
            return SharedMatrix(self.hidden_size, self.intermediate_size * (2 if self.hidden_act_gated else 1),
                                self.block_size, self.lowrank_dim)
        if key == "ffn_second":
            return SharedMatrix(self.intermediate_size, self.hidden_size, self.block_size, self.lowrank_dim)

        raise NotImplementedError(f"Unexpected SharedMatrix key: {key}")

    def init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, SharedMatrix):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, SemiSharedLinear):
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


class LeanTransformer(nn.Module):
    def __init__(self, config: LeanTransformerConfig):
        super().__init__()
        self.config = config
        self.layer_groups = nn.ModuleList([nn.ModuleDict(dict(layers=nn.ModuleList([nn.ModuleDict(dict(
                attention=self._make_attention(config), ffn=self._make_ffn(config)
        )) for _ in range(config.num_inner_groups)]))) for _ in range(config.num_hidden_groups)])
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self._sequential: Tuple[nn.Module, ...] = ()

    def _get_sequential(self):
        if not self._sequential:
            sequence = []
            for i in range(self.config.num_hidden_layers):
                group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
                for layer in self.layer_groups[group_idx].layers:
                    sequence.append(ActiveKwargs(layer.attention, ("attention_mask",), use_first_output=True))
                    sequence.append(ActiveKwargs(layer.ffn, active_keys=()))
            sequential_cls = ReversibleWithKwargs if self.config.reversible else SequentialWithKwargs
            self._sequential = (sequential_cls(*sequence), )
        return self._sequential[0]

    def _make_attention(self, config: LeanTransformerConfig):
        return LeanSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            attention_core=config.get_attention_core(),
            hidden_dropout_prob=config.hidden_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            dense_qkv=config.get_linear_layer("self_attn_qkv", config.hidden_size, config.hidden_size * 3, bias=True),
            dense_out=config.get_linear_layer("self_attn_out", config.hidden_size, config.hidden_size, bias=True),
            sandwich_norm=config.sandwich_norm,
            residual=not config.reversible, use_checkpoint=not config.reversible
        )

    def _make_ffn(self, config: LeanTransformerConfig):
        return LeanFFN(
            config.hidden_size,
            config.intermediate_size,
            activation=ACT2FN[config.hidden_act],
            gated=config.hidden_act_gated,
            layer_norm_eps=config.layer_norm_eps,
            dropout=config.hidden_dropout_prob,
            dense_i2h=config.get_linear_layer(
                "ffn_first", config.hidden_size, config.intermediate_size * (1 + config.hidden_act_gated), bias=True,
            ),
            dense_h2o=config.get_linear_layer("ffn_second", config.intermediate_size, config.hidden_size, bias=True),
            sandwich_norm=config.sandwich_norm,
            residual=not config.reversible,
        )

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self._get_sequential()(hidden_states, attention_mask=attention_mask)
        return BaseModelOutput(last_hidden_state=self.post_layer_norm(hidden_states))

    def init_weights(self):
        self.apply(self.config.init_weights)


class GradientCheckpointingMixin:
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool):
        if isinstance(module, LeanTransformer):
            sequential = module._get_sequential()
            assert not value or isinstance(sequential, SequentialWithKwargs), "Reversible does not need checkpoints"
            sequential.gradient_checkpointing = value
            for module in sequential:
                if isinstance(module, LeanSelfAttention):
                    # disable local checkpoints if checkpointing globally -- and vice versa
                    module.use_checkpoint = not value
