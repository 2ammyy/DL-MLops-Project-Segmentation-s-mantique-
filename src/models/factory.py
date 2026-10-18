import segmentation_models_pytorch as smp

MODELS = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "segformer": smp.Segformer,
}

DEFAULT_ENCODERS = {
    "unet": "resnet34",
    "unetplusplus": "resnet34",
    "deeplabv3plus": "resnet50",
    "segformer": "mit_b0",
}


def create_model(
    name="unet",
    encoder_name=None,
    encoder_weights="imagenet",
    classes=1,
    in_channels=3,
):
    key = name.lower().replace("-", "").replace("_", "")
    if key not in MODELS:
        raise ValueError(f"Modèle inconnu : {name}. Choix possibles : {sorted(MODELS)}")
    if encoder_name is None:
        encoder_name = DEFAULT_ENCODERS[key]
    if encoder_name.startswith("mit-"):
        encoder_name = encoder_name.replace("-", "_")
    return MODELS[key](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        in_channels=in_channels,
    )


def list_models():
    return sorted(MODELS)
