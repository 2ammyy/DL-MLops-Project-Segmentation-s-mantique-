from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.inference import SegmentationModel
from api.schemas import ErrorResponse, HealthResponse, PredictResponse

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = SegmentationModel()
    app.state.model_ready = True
    yield
    app.state.model_ready = False


app = FastAPI(
    title="PolypCopilot API",
    description="Segmentation de polypes colorectaux (Kvasir-SEG)",
    version="1.0.0",
    lifespan=lifespan,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    model: SegmentationModel = app.state.model
    return HealthResponse(
        status="ok",
        model_loaded=app.state.model_ready,
        checkpoint=model.checkpoint_path,
        val_dice=model.val_dice,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Fichier vide")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 10 Mo)")

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Image invalide ou format non supporte")

    try:
        result = app.state.model.predict(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Echec d'inference: {exc}") from exc

    return PredictResponse(filename=file.filename or "image", **result)
