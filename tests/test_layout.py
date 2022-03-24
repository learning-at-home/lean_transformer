import pytest
import torch
from lean_transformer import GeneralizedLinear, register_blocksparse_layout
from lean_transformer.models.albert import LeanAlbertConfig, LeanAlbertForPreTraining
import numpy as np


@pytest.mark.parametrize("layout,reference_sparsity", [
    (None, 1.0),
    ("pixelfly(block_size=4)", 0.5), ("pixelfly(block_size=2)", 0.3125), ("pixelfly(2, stretch=True)", 0.3125),
    ("hypercube(block_size=2, folded=False)", 0.3125), ("hypercube(block_size=2, folded=True)", 0.375),
    ("my_custom(foo=2)", 0.0625), ("my_custom(foo=4)", 0.125),
])
@pytest.mark.forked
def test_blocksparse_layout(layout: str, reference_sparsity: float):
    @register_blocksparse_layout("my_custom")
    def make_my_layout(out_features: int, in_features: int, foo: int) -> torch.BoolTensor:
        smaller_features = min(out_features, in_features)
        layout = torch.eye(smaller_features // foo, smaller_features // foo).to(torch.bool)
        layout = layout.repeat(out_features // smaller_features, in_features // smaller_features)
        return layout

    config = LeanAlbertConfig(vocab_size=1000, num_hidden_layers=4, hidden_size=32, num_attention_heads=4,
                              weight_layout=layout)
    model = LeanAlbertForPreTraining(config)

    batch = dict(
        input_ids=torch.tensor([
            [2, 339, 480,  60, 443,   9, 400,   3,   0,   0,   0,   0,   0,   0],
            [2, 339,  77, 257, 576, 202,  11, 417, 164, 881, 961, 631,   6,   3]]),
        attention_mask=torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))

    # test that model works
    out = model(**batch)

    if reference_sparsity is not None:
        sparsity_numerator = sparsity_denominator = 0

        for module in model.modules():
            if isinstance(module, GeneralizedLinear):
                sparsity_numerator += module.weight.numel()
                sparsity_denominator += module.out_features * module.in_features


        sparsity = sparsity_numerator / sparsity_denominator
        assert np.allclose(sparsity_numerator / sparsity_denominator, reference_sparsity), sparsity