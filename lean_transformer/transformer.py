from typing import Tuple

from torch import nn as nn
from transformers.modeling_outputs import BaseModelOutput

from lean_transformer import LeanFFN, LeanSelfAttention
from lean_transformer.config import LeanTransformerConfig
from lean_transformer.sequence import ActiveKwargs, ReversibleWithKwargs, SequentialWithKwargs


class LeanTransformer(nn.Module):
    """A generic transformer that does not hog your GPU memory; see gpt.py and albert.py for usage examples"""

    def __init__(self, config: LeanTransformerConfig):
        super().__init__()
        self.config = config
        self.layer_groups = []

        self.layer_groups = nn.ModuleList()
        for outer_index in range(config.num_hidden_groups):
            inner_group = nn.ModuleList([])
            for inner_index in range(config.num_inner_groups):
                index = outer_index * config.num_inner_groups + inner_index
                inner_group.append(nn.ModuleDict(dict(attention=self._make_attention(index, config),
                                                      ffn=self._make_ffn(index, config))))
            self.layer_groups.append(nn.ModuleDict(dict(layers=inner_group)))

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
            self._sequential = (sequential_cls(*sequence),)
        return self._sequential[0]

    def _make_attention(self, index: int, config: LeanTransformerConfig):
        return LeanSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            attention_core=config.get_attention_core(),
            dropout=config.hidden_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            dense_qkv=config.get_linear_layer(
                "self_attn_qkv", index, config.hidden_size, config.hidden_size * 3, bias=config.attn_qkv_bias),
            dense_out=config.get_linear_layer(
                "self_attn_out", index, config.hidden_size, config.hidden_size, bias=config.out_proj_bias),
            sandwich_norm=config.sandwich_norm,
            residual=not config.reversible, checkpoint_attention_core=not config.reversible
        )

    def _make_ffn(self, index: int, config: LeanTransformerConfig):
        return LeanFFN(
            config.hidden_size,
            config.intermediate_size,
            activation=self.config.get_activation_callable(),
            gated=config.hidden_act_gated,
            layer_norm_eps=config.layer_norm_eps,
            dropout=config.hidden_dropout_prob,
            dense_i2h=config.get_linear_layer("ffn_first", index, config.hidden_size,
                                              config.intermediate_size * (1 + config.hidden_act_gated), bias=True),
            dense_h2o=config.get_linear_layer("ffn_second", index, config.intermediate_size,
                                              config.hidden_size, bias=config.out_proj_bias),
            sandwich_norm=config.sandwich_norm,
            residual=not config.reversible,
        )

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self._get_sequential()(hidden_states, attention_mask=attention_mask)
        return BaseModelOutput(last_hidden_state=self.post_layer_norm(hidden_states))

    def init_weights(self):
        self.apply(self.config.init_weights)

    def gradient_checkpointing_enable(self, value: bool):
        sequential = self._get_sequential()
        assert not value or isinstance(sequential, SequentialWithKwargs), "Reversible does not need checkpoints"
        sequential.gradient_checkpointing = value
        for module in sequential:
            if isinstance(module, LeanSelfAttention):
                # disable local checkpoints if checkpointing globally -- and vice versa
                module.checkpoint_attention_core = not value


class GradientCheckpointingMixin:
    """A mix-in that enables gradient checkpoints in a huggingface model. See albert.py for usage examples."""

    supports_gradient_checkpointing: bool = True

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool):
        if isinstance(module, LeanTransformer):
            module.gradient_checkpointing_enable(value)
