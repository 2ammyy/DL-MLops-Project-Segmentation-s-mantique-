import pytest
import torch

from src.losses import BCEDiceLoss, DiceLoss, get_loss
from src.metrics import dice_score, evaluate_batch, iou_score


@pytest.fixture
def perfect_logits():
    targets = (torch.rand(2, 1, 64, 64) > 0.7).float()
    return targets * 20.0 - 10.0, targets


def test_dice_loss_perfect_prediction_near_zero(perfect_logits):
    logits, targets = perfect_logits
    assert DiceLoss()(logits, targets).item() < 0.01


def test_dice_loss_worst_when_inverted(perfect_logits):
    _, targets = perfect_logits
    worst = DiceLoss()(torch.zeros_like(targets), targets)
    right = DiceLoss()(targets * 20 - 10, targets)
    assert worst > right


def test_dice_loss_no_nan_on_empty_masks():
    logits = torch.zeros(1, 1, 32, 32)
    targets = torch.zeros(1, 1, 32, 32)
    loss = DiceLoss()(logits, targets)
    assert torch.isfinite(loss)


def test_bce_dice_combines_both_losses():
    logits = torch.randn(2, 1, 32, 32)
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
    combo = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)(logits, targets).item()
    dice = DiceLoss()(logits, targets).item()
    bce = torch.nn.BCEWithLogitsLoss()(logits, targets).item()
    assert abs(combo - (0.5 * dice + 0.5 * bce)) < 1e-5


def test_get_loss_valid_and_invalid():
    assert get_loss("dice") is not None
    assert get_loss("bce_dice") is not None
    assert get_loss("bce") is not None
    with pytest.raises(ValueError):
        get_loss("focal")


def test_metrics_perfect_prediction(perfect_logits):
    logits, targets = perfect_logits
    assert dice_score(logits, targets) > 0.99
    assert iou_score(logits, targets) > 0.99


def test_metrics_bounded(datasets):
    _, mask = datasets[0][0]
    logits = torch.randn(1, 1, 256, 256)
    dice = dice_score(logits, mask.unsqueeze(0))
    iou = iou_score(logits, mask.unsqueeze(0))
    assert 0.0 <= iou <= dice <= 1.0


def test_evaluate_batch_keys(perfect_logits):
    logits, targets = perfect_logits
    result = evaluate_batch(logits, targets)
    assert set(result) == {"dice", "iou"}
