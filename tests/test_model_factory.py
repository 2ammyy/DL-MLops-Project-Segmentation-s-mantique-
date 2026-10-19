import pytest
import torch

from src.models.factory import create_model, list_models


@pytest.mark.parametrize("name", ["unet", "unetplusplus", "deeplabv3plus", "segformer"])
def test_forward_output_shape(name):
    model = create_model(name, encoder_weights=None)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1, 256, 256)


def test_list_models():
    assert set(list_models()) == {"unet", "unetplusplus", "deeplabv3plus", "segformer"}


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        create_model("maskformer")


def test_mit_encoder_name_normalized():
    model = create_model("segformer", encoder_name="mit-b0", encoder_weights=None)
    assert model is not None


def test_encoder_weights_none_means_no_download():
    model = create_model("unet", encoder_name="resnet18", encoder_weights=None)
    assert sum(p.numel() for p in model.parameters()) > 0
