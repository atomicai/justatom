import torch

from justatom.training.loss import SoftContrastiveLoss


def test_soft_contrastive_loss_matches_expected_formula():
    loss_fn = SoftContrastiveLoss(margin=0.5, size_average=True)

    rep_anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    rep_other = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = loss_fn(rep_anchor, rep_other, labels)

    expected_positive = 0.5 * (0.0**2)
    expected_negative = 0.5 * max(0.0, 0.5 - 1.0) ** 2
    expected = (expected_positive + expected_negative) / 2.0

    assert torch.isclose(loss, torch.tensor(expected, dtype=torch.float32))
