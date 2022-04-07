from typing import Optional

import pytest
import torch

from lean_transformer.models.gpt import LeanGPTConfig, LeanGPTModel


@pytest.mark.parametrize("reversible, momentum_reversible_beta, dropout", [
    (False, None, 0), (False, None, 0.1), (True, None, 0), (False, None, 0.1), (True, 0.9, 0), (True, 0.1, 0), (True, 0.1, 0.1),
])
@pytest.mark.forked
def test_forward_backward_works(reversible: bool, momentum_reversible_beta: Optional[float], dropout: float):
    torch.use_deterministic_algorithms(True)
    config = LeanGPTConfig(
        vocab_size=1000, num_hidden_layers=4, hidden_size=32, num_attention_heads=4,
        hidden_dropout_prob=dropout, reversible=reversible, momentum_reversible_beta=momentum_reversible_beta)
    model = LeanGPTModel(config)
    batch = dict(
        input_ids=torch.tensor([
            [2, 339, 480, 60, 443, 9, 400, 3, 0, 0, 0, 0, 0, 0],
            [2, 339, 77, 257, 576, 202, 11, 417, 164, 881, 961, 631, 6, 3]]),
        attention_mask=torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))

    for i in range(2):
        out = model(**batch)
        out.logits.sum().backward()
