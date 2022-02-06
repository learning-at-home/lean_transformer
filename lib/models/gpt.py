# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch GPT modules that do not hog your GPU memory """

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from lib.models.transformer import GradientCheckpointingMixin, LeanTransformer, LeanTransformerConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LeanGPTConfig"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"


class LeanGPTConfig(LeanTransformerConfig):
    def __init__(
        self,
        *args,
        vocab_size: int = 50257,
        embedding_size: int = 1024,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs
    ):
        super().__init__(
            *args,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            type_vocab_size=type_vocab_size,
            tie_word_embeddings=True,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.type_vocab_size = type_vocab_size


class LeanGPTEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings. These embeddigns double as logits.
    """

    def __init__(self, config: LeanTransformerConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)

        self.token_type_embeddings = config.get_token_type_embeddings()
        self.position_embeddings = config.get_input_position_embeddings()

        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.embedding_size != config.hidden_size:
            self.embedding_hidden_mapping = nn.Linear(config.embedding_size, config.hidden_size)

        if self.position_embeddings is not None:
            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        if hasattr(self, "embedding_hidden_mapping"):
            embeddings = self.embedding_hidden_mapping(embeddings)
        return embeddings


class TiedMLMHead(nn.Module):
    def __init__(self, config, embeddings: LeanGPTEmbeddings):
        super().__init__()
        self.embeddings = embeddings

        if config.embedding_size != config.hidden_size:
            self.hidden_bias = nn.Parameter(torch.zeros(config.embedding_size))

        self.layer_norm = nn.LayerNorm(config.embedding_size)
        self.activation = ACT2FN[config.hidden_act]
        self.logits_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        if hasattr(self, "hidden_bias"):
            weight = self.embeddings.embedding_hidden_mapping.weight.t()
            hidden_states = F.linear(input=hidden_states, weight=weight, bias=self.hidden_bias)

        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = F.linear(input=hidden_states, weight=self.embeddings.word_embeddings.weight, bias=self.logits_bias)
        return logits


class LeanGPTForPreTraining(GradientCheckpointingMixin, PreTrainedModel):
    config_class = LeanGPTConfig
    base_model_prefix = "lean_gpt"

    def __init__(self, config: config_class):
        PreTrainedModel.__init__(self, config)

        self.config = config
        self.embeddings = LeanGPTEmbeddings(config)
        self.transformer = LeanTransformer(config)
        self.lm_head = TiedMLMHead(config, self.embeddings)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        assert isinstance(new_embeddings, nn.Embedding)
        self.embeddings.word_embeddings = new_embeddings
        prev_bias = self.lm_head.logits_bias
        intersection_size = min(len(prev_bias), new_embeddings.num_embeddings)
        self.lm_head.logits_bias = nn.Parameter(torch.zeros(new_embeddings.num_embeddings, dtype=prev_bias.dtype,
                                                            device=prev_bias.device, layout=prev_bias.layout))
        with torch.no_grad():
            self.lm_head.logits_bias[:intersection_size] = prev_bias[:intersection_size]

    def _init_weights(self, module: nn.Module):
        return self.config.init_weights(module)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert head_mask is None and output_attentions is None and output_hidden_states is None, "not implemented"
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        causal_attention_mask = torch.ones(seq_length, seq_length, dtype=self.dtype, device=device)
        causal_attention_mask = torch.tril(causal_attention_mask).view(1, 1, seq_length, seq_length)
        causal_attention_mask = (1.0 - causal_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        transformer_outputs = self.transformer(embedding_output, (extended_attention_mask, causal_attention_mask))
        lm_logits = self.lm_head(transformer_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            ignored_labels = torch.full_like(labels[..., :1], fill_value=-100)
            shift_labels = torch.cat([labels[..., 1:], ignored_labels], dim=1)
            loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.shape[-1]), shift_labels.view(-1), reduction="mean", ignore_index=-100
            )
            # note: masked labels have index -100 so they will be ignored when computing cross-entropy

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
