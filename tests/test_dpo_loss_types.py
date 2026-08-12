import math
import sys
import types

import pytest

from mv_audit.training.train_dpo import _preference_loss_values


class FakeTensor:
    def __init__(self, values):
        self.values = [float(value) for value in values]

    def __mul__(self, other):
        return FakeTensor([value * float(other) for value in self.values])

    __rmul__ = __mul__

    def __sub__(self, other):
        return FakeTensor([value - float(other) for value in self.values])

    def __neg__(self):
        return FakeTensor([-value for value in self.values])

    def pow(self, exponent):
        return FakeTensor([value**float(exponent) for value in self.values])


def _install_fake_torch(monkeypatch):
    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(
            functional=types.SimpleNamespace(
                logsigmoid=lambda tensor: FakeTensor(
                    [-math.log1p(math.exp(-value)) for value in tensor.values]
                )
            )
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def assert_values_close(actual: FakeTensor, expected: list[float]) -> None:
    assert actual.values == pytest.approx(expected)


def test_dpo_preference_loss_matches_logsigmoid(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    logits = FakeTensor([-1.0, 0.0, 2.0])
    beta = 0.1

    actual = _preference_loss_values(logits, beta=beta, loss_type="dpo")
    expected = [math.log1p(math.exp(-beta * value)) for value in logits.values]

    assert_values_close(actual, expected)


def test_ipo_preference_loss_targets_finite_margin(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    logits = FakeTensor([0.0, 5.0, 10.0])
    beta = 0.1

    actual = _preference_loss_values(logits, beta=beta, loss_type="ipo")

    assert_values_close(actual, [25.0, 0.0, 25.0])


def test_preference_loss_rejects_unknown_type(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported DPO loss_type"):
        _preference_loss_values(FakeTensor([0.0]), beta=0.1, loss_type="simpo")
