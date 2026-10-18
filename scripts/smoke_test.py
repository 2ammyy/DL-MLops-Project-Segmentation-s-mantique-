import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_fake_dataset(n=10):
    tmp = Path(tempfile.mkdtemp())
    images_dir = tmp / "images"
    masks_dir = tmp / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(n):
        img = rng.integers(0, 255, (100, 120, 3), dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        cv2.circle(mask, (60, 50), 25, 255, -1)
        cv2.imwrite(str(images_dir / f"img_{i}.jpg"), img)
        cv2.imwrite(str(masks_dir / f"img_{i}.jpg"), mask)
    return images_dir, masks_dir


def check(condition, label):
    if not condition:
        raise AssertionError(f"ECHEC : {label}")
    print(f"  [OK] {label}")


def main():
    print("== 1. Dataset & split ==")
    images_dir, masks_dir = make_fake_dataset()
    from src.data.dataset import get_datasets

    train_ds, val_ds, test_ds = get_datasets(images_dir, masks_dir, seed=42)
    check((len(train_ds), len(val_ds), len(test_ds)) == (7, 1, 2), "split 70/10/20")
    t2, v2, s2 = get_datasets(images_dir, masks_dir, seed=42)
    check(train_ds.pairs == t2.pairs and test_ds.pairs == s2.pairs, "split reproductible (seed)")

    x, y = train_ds[0]
    check(x.shape == (3, 256, 256) and x.dtype == torch.float32, "image (3,256,256) float32")
    check(y.shape == (1, 256, 256) and set(y.unique().tolist()) <= {0.0, 1.0}, "masque (1,256,256) binaire {0,1}")

    print("== 2. Modeles (forward) ==")
    from src.models.factory import create_model, list_models

    for name in list_models():
        model = create_model(name, encoder_weights=None)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 256, 256))
        check(out.shape == (2, 1, 256, 256), f"{name} forward -> (2,1,256,256)")

    print("== 3. Pertes & metriques ==")
    from src.losses import get_loss
    from src.metrics import dice_score, evaluate_batch, iou_score

    logits = torch.randn(2, 1, 256, 256)
    targets = (torch.rand(2, 1, 256, 256) > 0.7).float()
    for loss_name in ["dice", "bce_dice", "bce"]:
        value = get_loss(loss_name)(logits, targets)
        check(torch.isfinite(value), f"loss {loss_name} finite ({value.item():.4f})")

    perfect = evaluate_batch(targets * 20 - 10, targets)
    check(perfect["dice"] > 0.99 and perfect["iou"] > 0.99, "prediction parfaite -> Dice=1, IoU=1")
    check(iou_score(logits, targets) >= 0.0, "IoU aleatoire dans [0,1]")

    empty = dice_score(torch.full((1, 1, 64, 64), -10.0), torch.zeros(1, 1, 64, 64))
    check(abs(empty - 1.0) < 0.01, "masque vide vs vide -> Dice ~ 1 (pas NaN)")

    print("\nSMOKE TEST : TOUS LES TESTS PASSENT")


if __name__ == "__main__":
    main()
