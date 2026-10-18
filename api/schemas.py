from pydantic import BaseModel, ConfigDict, Field


class PolypBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class PredictResponse(PolypBaseModel):
    filename: str
    original_width: int
    original_height: int
    mask_base64: str = Field(description="Masque PNG encode en base64, taille originale")
    polyp_area_ratio: float = Field(ge=0.0, le=1.0, description="Surface du polype / image")
    inference_ms: float
    model_version: str


class HealthResponse(PolypBaseModel):
    status: str
    model_loaded: bool
    checkpoint: str
    val_dice: float


class ErrorResponse(BaseModel):
    detail: str
