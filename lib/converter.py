from typing import Dict

import torch


def untie_embeddings(state: Dict[str, torch.Tensor]):
    state['lm_head.weight'] = state['embeddings.embedding_hidden_mapping.weight'].clone().t()
