import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "pipelines" / "baseline.json"
CANDIDATE = ROOT / "models" / "registry" / "metrics.json"


def promote(candidate_path=None, baseline_path=None):
    candidate_path = Path(candidate_path) if candidate_path else CANDIDATE
    baseline_path = Path(baseline_path) if baseline_path else BASELINE

    with open(candidate_path, encoding="utf-8") as f:
        candidate = json.load(f)

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(candidate_path, baseline_path)
    print(
        f"Baseline mise a jour : Dice={candidate['dice']:.4f} "
        f"IoU={candidate['iou']:.4f} -> {baseline_path}"
    )


if __name__ == "__main__":
    promote(*(sys.argv[1:3] or []))
