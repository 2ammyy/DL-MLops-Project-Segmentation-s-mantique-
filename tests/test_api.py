import base64

import cv2
import numpy as np


def test_health(api_client):
    res = api_client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_mask(api_client, sample_image):
    res = api_client.post("/predict", files={"file": ("img.jpg", sample_image, "image/jpeg")})
    assert res.status_code == 200
    body = res.json()

    raw = base64.b64decode(body["mask_base64"])
    nparr = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    assert mask is not None
    assert mask.shape == (body["original_height"], body["original_width"])
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert 0.0 <= body["polyp_area_ratio"] <= 1.0
    assert body["model_version"]


def test_predict_invalid_file(api_client):
    res = api_client.post(
        "/predict", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")}
    )
    assert res.status_code == 400


def test_predict_empty_file(api_client):
    res = api_client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert res.status_code == 400
