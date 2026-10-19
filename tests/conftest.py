import os
from pathlib import Path

import numpy as np
import pytest

from api.inference import SegmentationModel
from src.data.dataset import get_datasets

ROOT = Path(__file__).resolve().parents[1]
REAL_CHECKPOINT = ROOT / "models" / "checkpoints" / "best.pt"


@pytest.fixture(scope="session")
def fake_data_dir(tmp_path_factory):
    import cv2

    base = tmp_path_factory.mktemp("kvasir_fake")
    images_dir = base / "images"
    masks_dir = base / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(10):
        img = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
        mask = np.zeros((96, 128), dtype=np.uint8)
        cv2.circle(mask, (64, 48), 20, 255, -1)
        cv2.imwrite(str(images_dir / f"img_{i}.jpg"), img)
        cv2.imwrite(str(masks_dir / f"img_{i}.jpg"), mask)
    return {"images": images_dir, "masks": masks_dir}


@pytest.fixture(scope="session")
def datasets(fake_data_dir):
    return get_datasets(fake_data_dir["images"], fake_data_dir["masks"], seed=42)


@pytest.fixture(scope="session")
def api_client(tmp_path_factory):
    if REAL_CHECKPOINT.exists():
        checkpoint = REAL_CHECKPOINT
    else:
        import torch

        from src.models.factory import create_model

        checkpoint = tmp_path_factory.mktemp("ckpt") / "toy.pt"
        model = create_model("unet", encoder_name="resnet18", encoder_weights=None)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": 0,
                "best_dice": 0.0,
                "model": "unet",
                "encoder": "resnet18",
                "encoder_weights": "none",
                "loss": "bce_dice",
                "image_size": 256,
                "seed": 42,
                "metrics": {},
            },
            checkpoint,
        )
    os.environ["CHECKPOINT_PATH"] = str(checkpoint)
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def sample_image(fake_data_dir):
    import cv2

    path = sorted(fake_data_dir["images"].glob("*.jpg"))[0]
    ok, buf = cv2.imencode(".jpg", cv2.imread(str(path)))
    assert ok
    return buf.tobytes()
