from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.rag_chatbot import ask_question

from api.inference import SegmentationModel
from api.metrics import MODEL_DICE_GAUGE, PREDICTION_COUNT
from api.schemas import ErrorResponse, HealthResponse, PredictResponse

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = SegmentationModel()
    app.state.model_ready = True
    MODEL_DICE_GAUGE.set(app.state.model.val_dice)
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
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Echec d'inference: {exc}") from exc

    PREDICTION_COUNT.labels(status="success").inc()
    return PredictResponse(filename=file.filename or "image", **result)


@app.get("/metrics")
async def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint du chatbot médical RAG.
    Reçoit une question et retourne une réponse basée sur les guidelines.
    """
    result = ask_question(request.message)
    return result

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
