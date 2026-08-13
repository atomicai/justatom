from __future__ import annotations

import pytest
import torch

from justatom.training.gradient_projection import project_conflicting_gradients


def test_conflicting_memory_gradient_is_projected_orthogonally():
    primary = [torch.tensor([1.0, 0.0])]
    memory = [torch.tensor([-1.0, 2.0])]

    projected, stats = project_conflicting_gradients(primary, memory)

    assert projected[0] is not None
    torch.testing.assert_close(projected[0], torch.tensor([0.0, 2.0]))
    assert torch.dot(primary[0], projected[0]).item() == pytest.approx(0.0, abs=1e-7)
    assert stats.conflict is True
    assert stats.cosine < 0.0
    assert stats.coefficient == pytest.approx(-1.0)


def test_aligned_memory_gradient_is_unchanged():
    primary = [torch.tensor([1.0, 0.0])]
    memory = [torch.tensor([2.0, 1.0])]

    projected, stats = project_conflicting_gradients(primary, memory)

    torch.testing.assert_close(projected[0], memory[0])
    assert stats.conflict is False
    assert stats.coefficient == 0.0


def test_memory_only_parameter_gradient_is_retained():
    primary = [torch.tensor([1.0]), None]
    memory = [torch.tensor([-1.0]), torch.tensor([3.0])]

    projected, stats = project_conflicting_gradients(primary, memory)

    torch.testing.assert_close(projected[0], torch.zeros(1))
    torch.testing.assert_close(projected[1], torch.tensor([3.0]))
    assert stats.conflict is True


def test_projection_spans_primary_only_parameters():
    primary = [torch.tensor([1.0]), torch.tensor([2.0])]
    memory = [torch.tensor([-1.0]), None]

    projected, stats = project_conflicting_gradients(primary, memory)

    assert projected[0] is not None and projected[1] is not None
    torch.testing.assert_close(projected[0], torch.tensor([-0.8]))
    torch.testing.assert_close(projected[1], torch.tensor([0.4]))
    projected_dot = sum(torch.dot(gp, gm).item() for gp, gm in zip(primary, projected) if gm is not None)
    assert projected_dot == pytest.approx(0.0, abs=1e-7)
    assert stats.conflict is True


def test_projection_rejects_mismatched_gradient_lists():
    with pytest.raises(ValueError, match="equal length"):
        project_conflicting_gradients([torch.ones(1)], [])
