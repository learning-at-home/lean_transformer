from functools import partial
from typing import Tuple, Optional, Union

from torch import nn as nn
from torch.autograd.graph import saved_tensors_hooks
from transformers import PreTrainedModel
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

    def configure_optimizations(
            self, *,
            gradient_checkpointing: Optional[Union[bool, int]] = None,
            checkpoint_last: Optional[bool] = None,
            checkpoint_hook: Optional[saved_tensors_hooks] = None,
            preserve_rng_state: Optional[bool] = None,
            checkpoint_attention_core: Optional[bool] = None,
            ffn_custom_grad: Optional[bool] = None,
    ):
        """
        Set one or more memory saving options for all compatible sub-modules. Options set to None remain unchanged.
        Unlike main config, optimizatons do not affect model predictions, changing only memory/compute trade-offs.

        :param gradient_checkpointing: configure gradient checkpointing for non-reversible models, True/False or integer
          - if False, disable layer-wise gradient checkpoints (default)
          - if True, apply checkpoints to each individual layer, attention and mlp are separate layers
          - if an integer, use spread this many checkpoints evenly across all layers, similar to checkpoint_sequential
        :param checkpoint_last: if True, checkpoints the last chunk of layers the same way as all other layers.
           If False, does not apply checkpointing to last layers, which is faster but may cause OOM when computing loss
        :param checkpoint_hook: optionally compress gradient checkpoints with a user-defined autograd hook
           See https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html for details.
        :param preserve_rng_state: enable or disable saving RNG state for checkpoints or reversible layers.
          Setting this to False will slightly improve speed and memory if there is no randomness such as dropout.
          **Warning** if layers do contain randomness (e.g. dropout), save_rng_state=False may cause incorrect backprop

        :param checkpoint_attention_core: re-compute attention weights during backward pass instead of storing them
        :param ffn_custom_grad: use manual FFN backprop that saves memory at the cost of a little extra compute time

        """

        sequential = self._get_sequential()
        if isinstance(sequential, ReversibleWithKwargs):
            if gradient_checkpointing or checkpoint_hook or checkpoint_last:
                raise ValueError("Reversible does not support gradient checkpointing")
            sequential.preserve_rng_state = preserve_rng_state

        else:
            assert isinstance(sequential, SequentialWithKwargs)
            sequential.configure_gradient_checkpointing(
                gradient_checkpointing, checkpoint_last, checkpoint_hook, preserve_rng_state
            )

        for module in sequential:
            if checkpoint_attention_core is not None and isinstance(module, LeanSelfAttention):
                module.checkpoint_attention_core = checkpoint_attention_core
            elif ffn_custom_grad is not None and isinstance(module, LeanFFN):
                module.custom_grad = ffn_custom_grad
            else:
                # if this fails, you need to make sure that optimizations are propagated to new layers
                assert not hasattr(module, "checkpoint_attention_core")
                assert not hasattr(module, "custom_grad")


class OptimizationsMixin(PreTrainedModel):
    """adds set_optimizations to any huggingface model that involves LeanTransformer. See example in albert.py"""

    def set_optimizations(self, *args, **kwargs):
        """find any LeanTransformer sub-modules and pass settings to them. See LeanTransformer.set_optimizations """
        if not any(isinstance(module, LeanTransformer) for module in self.modules()):
            raise ValueError("Cannot set_optimizations: no LeanTransformer sub-modules found")
        self.apply(partial(self._set_optimizations, *args, **kwargs))

    @staticmethod
    def _set_optimizations(module: nn.Module, *args, **kwargs):
        if isinstance(module, LeanTransformer):
            module.set_optimizations(*args, **kwargs)

    supports_gradient_checkpointing: bool = True

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool):
        """compatibility with hugging face gradient checkpointing"""
        if isinstance(module, LeanTransformer):
            module.set_optimizations(gradient_checkpointing=value)
