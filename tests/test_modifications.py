from contextlib import nullcontext
from typing import Union, Optional

import pytest
import torch

from lean_transformer.models.albert import LeanAlbertConfig, LeanAlbertForPreTraining


@pytest.mark.parametrize("dropout,preserve_rng,grad_fails", [(0.1, None, False)])
@pytest.mark.parametrize(
    "reversible,checkpoints,checkpoint_last,custom_attn,custom_ffn",
    [(False, False, False, False, False), (False, False, False, True, False), (False, False, False, False, True),
     (True, False, False, False, False), (True, False, False, True, True),
     (False, True, False, True, False), (False, True, False, False, True), (False, True, True, False, True),
     (False, 4, True, True, True), (False, 4, False, True, True), (False, 8, False, True, True),
     (False, 1, False, True, True), (False, 3, False, True, True), (False, 3, False, True, False)])
@pytest.mark.forked
def test_modification_consistency(
        reversible, checkpoints, checkpoint_last, custom_attn, custom_ffn, preserve_rng, dropout, grad_fails):
    return _test_modification_consistency(
        reversible, checkpoints, checkpoint_last, custom_attn, custom_ffn, preserve_rng, dropout, grad_fails)


@pytest.mark.parametrize("dropout,preserve_rng,grad_fails", [
    (0.1, None, False), (0.1, True, False), (0.0, False, False), (0.1, False, True)])
@pytest.mark.parametrize("reversible,checkpoints,checkpoint_last,custom_attn,custom_ffn", [(False, True, True, False, False)])
@pytest.mark.forked
def test_preserve_rng(
        reversible, checkpoints, checkpoint_last, custom_attn, custom_ffn, preserve_rng, dropout, grad_fails):
    return _test_modification_consistency(
        reversible, checkpoints, checkpoint_last, custom_attn, custom_ffn, preserve_rng, dropout, grad_fails)


def _test_modification_consistency(
        reversible: bool, checkpoints: Union[bool, int], checkpoint_last: bool, custom_attn: bool, custom_ffn: bool,
        preserve_rng: Optional[bool], dropout: float, grad_fails: bool):
    torch.use_deterministic_algorithms(True)
    config = LeanAlbertConfig(
        vocab_size=1000, num_hidden_layers=8, hidden_size=64, num_attention_heads=8,
        hidden_dropout_prob=dropout, reversible=reversible)
    model = LeanAlbertForPreTraining(config)

    batch = dict(
        input_ids=torch.tensor([
            [2, 339, 480,  60, 443,   9, 400,   3,   0,   0,   0,   0,   0,   0],
            [2, 339,  77, 257, 576, 202,  11, 417, 164, 881, 961, 631,   6,   3]]),
        attention_mask=torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))

    for i in range(2):
        model.set_optimizations(
            gradient_checkpointing=False, checkpoint_last=False,
            checkpoint_attention_core=False, ffn_custom_grad=False)
        torch.manual_seed(1337)
        out = model(**batch)
        out.prediction_logits.sum().backward()
        ref_logits = out.prediction_logits.detach().clone()
        ref_grads = [param.grad.detach().clone() for param in model.albert.transformer.parameters()]
        model.zero_grad(set_to_none=True)

    model.set_optimizations(
        gradient_checkpointing=checkpoints, checkpoint_last=checkpoint_last,
        checkpoint_attention_core=custom_attn, ffn_custom_grad=custom_ffn, preserve_rng_state=preserve_rng)
    torch.manual_seed(1337)
    out = model(**batch)
    out.prediction_logits.sum().backward()
    our_logits = out.prediction_logits.detach().clone()
    our_grads = [param.grad.detach().clone() for param in model.albert.transformer.parameters()]
    model.zero_grad(set_to_none=True)
    assert torch.allclose(ref_logits, our_logits, rtol=0, atol=1e-5), abs(ref_logits-our_logits).max()

    with pytest.raises(AssertionError) if grad_fails else nullcontext():
        for g_ref, g_our in zip(ref_grads, our_grads):
            assert torch.allclose(g_ref, g_our, rtol=0, atol=1e-5), abs(g_ref-g_our).max()
