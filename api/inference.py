import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.augmentation import get_val_transforms
from src.models.factory import create_model


class SegmentationModel:
    def __init__(self, checkpoint_path=None, device=None, threshold=0.5):
        if checkpoint_path is None:
            checkpoint_path = os.environ.get(
                "CHECKPOINT_PATH", "models/checkpoints/best.pt"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = threshold
        self.checkpoint_path = str(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = create_model(
            name=checkpoint["model"],
            encoder_name=checkpoint.get("encoder"),
            encoder_weights=None,
            classes=1,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_val_transforms()
        self.version = (
            f"{checkpoint['model']}-{checkpoint.get('encoder') or 'default'}"
            f"-ep{checkpoint['epoch'] + 1}"
        )
        self.val_dice = float(checkpoint.get("best_dice", 0.0))

    @torch.no_grad()
    def predict(self, image_bgr):
        start = time.perf_counter()
        h0, w0 = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=rgb)["image"].unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        mask_small = (torch.sigmoid(logits)[0, 0] >= self.threshold).cpu().numpy().astype(np.uint8)

        mask_full = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
        area_ratio = float(mask_full.mean())
        ok, buf = cv2.imencode(".png", mask_full * 255)
        if not ok:
            raise RuntimeError("Echec encodage PNG du masque")
        mask_base64 = base64.b64encode(buf.tobytes()).decode("ascii")

        return {
            "mask_base64": mask_base64,
            "original_width": w0,
            "original_height": h0,
            "polyp_area_ratio": area_ratio,
            "inference_ms": round((time.perf_counter() - start) * 1000, 1),
            "model_version": self.version,
        }
