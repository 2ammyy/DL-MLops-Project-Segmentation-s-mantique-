import torch

DIMS = (1, 2, 3)


def _binarize(logits, targets, threshold):
    preds = (torch.sigmoid(logits) >= threshold).to(targets.dtype)
    targets = (targets >= 0.5).to(targets.dtype)
    return preds, targets


@torch.no_grad()
def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds, targets = _binarize(logits, targets, threshold)
    intersection = (preds * targets).sum(dim=DIMS)
    union = preds.sum(dim=DIMS) + targets.sum(dim=DIMS) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


@torch.no_grad()
def dice_score(logits, targets, threshold=0.5, smooth=1.0):
    preds, targets = _binarize(logits, targets, threshold)
    intersection = (preds * targets).sum(dim=DIMS)
    cardinality = preds.sum(dim=DIMS) + targets.sum(dim=DIMS)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice.mean().item()


@torch.no_grad()
def evaluate_batch(logits, targets, threshold=0.5):
    return {
        "dice": dice_score(logits, targets, threshold),
        "iou": iou_score(logits, targets, threshold),
    }
