import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import get_datasets
from src.metrics import dice_score, iou_score
from src.models.factory import create_model

DEFAULT_CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "models/checkpoints/best.pt")
DEFAULT_SAMPLES = int(os.environ.get("NR_SAMPLE_SIZE", "32"))
DEFAULT_SEED = int(os.environ.get("NR_SEED", "42"))
DEFAULT_OUT = os.environ.get("CANDIDATE_PATH", "models/registry/metrics.json")
DATA_DIR = ROOT / "data" / "raw" / "Kvasir-SEG"


def ensure_dataset():
    if DATA_DIR.exists() and (DATA_DIR / "images").exists():
        return
    raise SystemExit(
        "Dataset absent : lancez python scripts/download_dataset.py d'abord"
    )


def evaluate(checkpoint_path=DEFAULT_CHECKPOINT, samples=DEFAULT_SAMPLES,
             seed=DEFAULT_SEED, out_path=DEFAULT_OUT):
    ensure_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ensure_device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(
        name=checkpoint["model"],
        encoder_name=checkpoint.get("encoder"),
        encoder_weights=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_ds = get_datasets(DATA_DIR / "images", DATA_DIR / "masks", seed=seed)
    n = min(samples, len(test_ds))
    subset = torch.utils.data.Subset(test_ds, range(n))

    dice_values, iou_values = [], []
    with torch.no_grad():
        for i in range(0, n, 8):
            batch = [subset[j] for j in range(i, min(i + 8, n))]
            images = torch.stack([b[0] for b in batch]).to(device)
            masks = torch.stack([b[1] for b in batch]).to(device)
            logits = model(images)
            for k in range(images.size(0)):
                dice_values.append(dice_score(logits[k:k + 1], masks[k:k + 1]))
                iou_values.append(iou_score(logits[k:k + 1], masks[k:k + 1]))

    metrics = {
        "dice": sum(dice_values) / len(dice_values),
        "iou": sum(iou_values) / len(iou_values),
        "samples": n,
        "seed": seed,
        "model": checkpoint["model"],
        "encoder": checkpoint.get("encoder"),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_path": str(checkpoint_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get("MLFLOW_RUN_ID", "local"),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
