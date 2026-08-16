from __future__ import annotations

import pytest
import torch

from justatom.training.auxiliary_gradient import control_auxiliary_gradients
from justatom.training.config import AuxiliaryGradientMode


def test_safe_gradient_is_cosine_compatible_and_norm_capped():
    primary = [torch.tensor([1.0, 0.0])]
    auxiliary = [torch.tensor([2.0, 0.0])]

    controlled, stats = control_auxiliary_gradients(
        primary,
        auxiliary,
        mode=AuxiliaryGradientMode.SAFE,
        max_norm_ratio=0.25,
        eps=1e-12,
    )

    torch.testing.assert_close(controlled[0], torch.tensor([0.25, 0.0]))
    assert stats.cosine_scale == pytest.approx(1.0)
    assert stats.norm_scale == pytest.approx(0.125)
    assert stats.total_scale == pytest.approx(0.125)
    direction = primary[0] + controlled[0]
    assert torch.dot(primary[0], direction) >= torch.dot(primary[0], primary[0])
    assert controlled[0].norm() <= 0.25 * primary[0].norm()


def test_safe_gradient_uses_shared_dot_and_complete_parameter_norms():
    primary = [torch.tensor([1.0]), None]
    auxiliary = [torch.tensor([2.0]), torch.tensor([3.0])]

    controlled, stats = control_auxiliary_gradients(
        primary,
        auxiliary,
        mode=AuxiliaryGradientMode.SAFE,
        max_norm_ratio=1.0,
        eps=1e-12,
    )

    expected_cosine = 2.0 / (13.0**0.5)
    expected_total_scale = expected_cosine / (2.0 + 1e-12)
    assert stats.dot == pytest.approx(2.0)
    assert stats.cosine == pytest.approx(expected_cosine)
    assert stats.total_scale == pytest.approx(expected_total_scale)
    torch.testing.assert_close(controlled[0], torch.tensor([2.0 * expected_total_scale]))
    torch.testing.assert_close(controlled[1], torch.tensor([3.0 * expected_total_scale]))


@pytest.mark.parametrize(
    "auxiliary",
    [torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])],
    ids=["orthogonal", "conflicting"],
)
def test_safe_gradient_suppresses_non_positive_shared_dot(auxiliary):
    controlled, stats = control_auxiliary_gradients(
        [torch.tensor([1.0, 0.0])],
        [auxiliary],
        mode=AuxiliaryGradientMode.SAFE,
        max_norm_ratio=0.25,
        eps=1e-12,
    )

    torch.testing.assert_close(controlled[0], torch.zeros_like(auxiliary))
    assert stats.cosine_scale == 0.0
    assert stats.norm_scale == 0.0
    assert stats.total_scale == 0.0
    assert stats.compatible is False


@pytest.mark.parametrize(
    ("primary", "auxiliary"),
    [
        ([torch.zeros(2)], [torch.tensor([1.0, 2.0])]),
        ([torch.tensor([1.0, 2.0])], [torch.zeros(2)]),
    ],
    ids=["zero-primary", "zero-auxiliary"],
)
def test_safe_gradient_suppresses_zero_norm_gradient(primary, auxiliary):
    controlled, stats = control_auxiliary_gradients(
        primary,
        auxiliary,
        mode=AuxiliaryGradientMode.SAFE,
        max_norm_ratio=0.25,
        eps=1e-12,
    )

    torch.testing.assert_close(controlled[0], torch.zeros_like(auxiliary[0]))
    assert stats.cosine == 0.0
    assert stats.total_scale == 0.0


def test_observe_returns_auxiliary_gradient_unchanged_and_reports_metrics():
    primary = [torch.tensor([1.0, 0.0])]
    auxiliary = [torch.tensor([2.0, 1.0])]

    controlled, stats = control_auxiliary_gradients(
        primary,
        auxiliary,
        mode=AuxiliaryGradientMode.OBSERVE,
        max_norm_ratio=0.25,
        eps=1e-12,
    )

    torch.testing.assert_close(controlled[0], auxiliary[0])
    assert controlled[0] is not auxiliary[0]
    assert stats.cosine_scale == 1.0
    assert stats.norm_scale == 1.0
    assert stats.total_scale == 1.0
    assert set(stats.metrics()) == {
        "gradient/retrieval_norm",
        "gradient/auxiliary_norm",
        "gradient/auxiliary_controlled_norm",
        "gradient/auxiliary_dot",
        "gradient/auxiliary_cosine",
        "gradient/auxiliary_compatible",
        "gradient/auxiliary_cosine_scale",
        "gradient/auxiliary_norm_scale",
        "gradient/auxiliary_total_scale",
    }


def test_off_mode_returns_auxiliary_gradient_unchanged():
    auxiliary = [torch.tensor([2.0, -1.0]), None]

    controlled, stats = control_auxiliary_gradients(
        [torch.tensor([1.0, 0.0]), None],
        auxiliary,
        mode=AuxiliaryGradientMode.OFF,
        max_norm_ratio=0.0,
        eps=1e-12,
    )

    torch.testing.assert_close(controlled[0], auxiliary[0])
    assert controlled[1] is None
    assert stats.total_scale == 1.0


def test_observe_empty_gradient_lists_report_pass_through_scales():
    controlled, stats = control_auxiliary_gradients(
        [None],
        [None],
        mode=AuxiliaryGradientMode.OBSERVE,
        max_norm_ratio=0.25,
        eps=1e-12,
    )

    assert controlled == [None]
    assert stats.cosine_scale == 1.0
    assert stats.norm_scale == 1.0
    assert stats.total_scale == 1.0


def test_controller_preserves_none_entries_and_rejects_mismatched_lists():
    with pytest.raises(ValueError, match="equal length"):
        control_auxiliary_gradients(
            [torch.ones(1)],
            [],
            mode=AuxiliaryGradientMode.SAFE,
            max_norm_ratio=0.25,
            eps=1e-12,
        )

    controlled, _ = control_auxiliary_gradients(
        [None, torch.tensor([1.0])],
        [None, torch.tensor([2.0])],
        mode=AuxiliaryGradientMode.SAFE,
        max_norm_ratio=0.25,
        eps=1e-12,
    )
    assert controlled[0] is None
    assert controlled[1] is not None


@pytest.mark.parametrize(
    "primary, auxiliary",
    [
        ([torch.tensor([float("nan")])], [torch.ones(1)]),
        ([torch.ones(1)], [torch.tensor([float("inf")])]),
    ],
    ids=["non-finite-primary", "non-finite-auxiliary"],
)
def test_controller_rejects_non_finite_aggregate_statistics(primary, auxiliary):
    with pytest.raises(ValueError, match="non-finite auxiliary gradient statistics"):
        control_auxiliary_gradients(
            primary,
            auxiliary,
            mode=AuxiliaryGradientMode.SAFE,
            max_norm_ratio=0.25,
            eps=1e-12,
        )
