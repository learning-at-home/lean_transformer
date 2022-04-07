from itertools import chain

import torch
import torch.nn as nn
import revlib

from lean_transformer.sequence import MeanCoupling, MeanReversibleWithKwargs, ActiveKwargs


def test_mean_coupling():
    couplings = [MeanCoupling(i + 1) for i in range(3)]
    x0 = torch.ones(3)
    x1 = couplings[0].forward(x0, x0 * 0 + 3)
    x2 = couplings[1].forward(x1, x1 * 0 + 6)
    x3 = couplings[2].forward(x2, x2 * 0 - 4)

    assert torch.allclose(x1, torch.full_like(x1, (1 + 3) / 2))
    assert torch.allclose(x2, torch.full_like(x1, (1 + 3 + 6) / 3))
    assert torch.allclose(x3, torch.full_like(x1, (1 + 3 + 6 - 4) / 4))

    x2_rev = couplings[2].inverse(x3, x0 * 0 - 4)
    x1_rev = couplings[1].inverse(x2_rev, x0 * 0 + 6)
    x0_rev = couplings[0].inverse(x1_rev, x0 * 0 + 3)

    assert torch.allclose(x2, x2_rev)
    assert torch.allclose(x1, x1_rev)
    assert torch.allclose(x0, x0_rev)


class DummyLayer(nn.Linear):
    def __init__(self, c):
        super().__init__(3, 3, bias=True)
        self.weight.data *= 0
        self.bias.data[:] = c


class Identity(nn.Sequential):
    pass


def test_mean_reversible_naive():
    model = revlib.ReversibleSequential(
        DummyLayer(1), Identity(), DummyLayer(3), Identity(), DummyLayer(14), Identity(), DummyLayer(-5), Identity(),
        coupling_forward=list(chain(*zip([MeanCoupling(i + 1).forward for i in range(4)],
                                         [revlib.additive_coupling_forward] * 4))),
        coupling_inverse=list(chain(*zip([MeanCoupling(i + 1).inverse for i in range(4)],
                                         [revlib.additive_coupling_inverse] * 4)))
    )

    inp = torch.zeros(3) - 2
    inp = inp.detach().requires_grad_(True)
    zeros = torch.zeros(3)
    out = (inp, inp, zeros, zeros)

    for rev_layer in model.children():
        out = rev_layer(out)

    assert torch.allclose(out[0], zeros + (-2 + 1 + 3 + 14 - 5) / 5)
    assert torch.allclose(out[1], zeros + (
            (-2 + 1 + 3 + 14 - 5) / 5 +
            (-2 + 1 + 3 + 14) / 4 +
            (-2 + 1 + 3) / 3 +
            (-2 + 1) / 2 +
            (-2) / 1
    ))

    out[0].mean().backward()  # just check that backward does not fail
    assert torch.allclose(inp.grad, zeros + 1 / 3)


def test_mean_reversible():
    model = MeanReversibleWithKwargs(*(
        ActiveKwargs(m, ()) for m in (DummyLayer(1), DummyLayer(3), DummyLayer(14), DummyLayer(-5))
    ))
    inp = torch.zeros(4, 3) - 2
    inp = inp.detach().requires_grad_(True)
    zeros = torch.zeros_like(inp)
    out = model(inp)
    out.mean().backward()

    assert torch.allclose(out, zeros + (-2 + 1 + 3 + 14 - 5) / 5)
    assert torch.allclose(inp.grad, zeros + 1 / 12)
