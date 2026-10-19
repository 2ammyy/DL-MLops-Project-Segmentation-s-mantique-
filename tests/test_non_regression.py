import json
import os
from pathlib import Path

import pytest

THRESHOLD = 0.01
BASELINE_PATH = Path(os.environ.get("BASELINE_PATH", "pipelines/baseline.json"))
CANDIDATE_PATH = Path(os.environ.get("CANDIDATE_PATH", "models/registry/metrics.json"))


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="baseline absente")
def test_baseline_has_required_fields():
    baseline = _load(BASELINE_PATH)
    for field in ["dice", "iou", "model", "run_id", "created_at"]:
        assert field in baseline, f"champ manquant dans la baseline : {field}"


@pytest.mark.skipif(
    not CANDIDATE_PATH.exists(),
    reason="pas de candidat (lance python scripts/evaluate.py apres un entranement)",
)
def test_dice_non_regression():
    baseline = _load(BASELINE_PATH)
    candidate = _load(CANDIDATE_PATH)
    dice_v1 = baseline["dice"]
    dice_v2 = candidate["dice"]
    min_accepted = dice_v1 - THRESHOLD
    assert dice_v2 >= min_accepted, (
        f"NON-REGRESSION DETECTEE : Dice_v2={dice_v2:.4f} < "
        f"Dice_v1={dice_v1:.4f} - {THRESHOLD} = {min_accepted:.4f}"
    )


@pytest.mark.skipif(
    not CANDIDATE_PATH.exists(),
    reason="pas de candidat (lance python scripts/evaluate.py apres un entraonnement)",
)
def test_iou_non_regression():
    baseline = _load(BASELINE_PATH)
    candidate = _load(CANDIDATE_PATH)
    iou_v1 = baseline["iou"]
    iou_v2 = candidate["iou"]
    assert iou_v2 >= iou_v1 - THRESHOLD, (
        f"NON-REGRESSION IoU : IoU_v2={iou_v2:.4f} < IoU_v1={iou_v1:.4f} - {THRESHOLD}"
    )
