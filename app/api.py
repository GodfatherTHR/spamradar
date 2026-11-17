from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.inference import SpamPredictor
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spamradar.api")

ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"
CONFIG = load_config()
MAX_BATCH = int(CONFIG.get("api", {}).get("max_batch_size", 256))
MAX_TEXT = int(CONFIG.get("api", {}).get("max_text_length", 4000))

runtime: dict[str, Any] = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading SpamRadar predictor...")
    try:
        runtime["predictor"] = SpamPredictor(warmup=True)
        runtime["error"] = None
        log.info("Predictor ready: %s", runtime["predictor"].health())
    except Exception as exc:
        runtime["predictor"] = None
        runtime["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("Failed to load predictor")
    yield
    runtime["predictor"] = None


app = FastAPI(
    title="SpamRadar Multilingual API",
    description=(
        "Multilingual spam and social-engineering detection with calibrated risk tiers, "
        "engineered signals, and model explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class PredictRequest(BaseModel):
    text: str = Field(..., description="Message text to analyze", max_length=MAX_TEXT)
    language: str | None = Field(default=None, description="Optional language hint")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be empty")
        return value


class BatchPredictRequest(BaseModel):
    messages: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)

    @field_validator("messages")
    @classmethod
    def messages_not_blank(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values if v and v.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty message is required")
        if len(cleaned) > MAX_BATCH:
            raise ValueError(f"At most {MAX_BATCH} messages are allowed")
        return cleaned


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    label: str
    risk_tier: str
    spam_probability: float
    thresholds: dict[str, float]
    language: str
    language_confidence: float
    detected_signals: list[dict]
    model_cues: list[dict]
    semantic_match: dict | None = None
    scores: dict
    latency_ms: float
    text_length: int
    explanation: str
    model_version: str


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
    summary: dict
    latency_ms: float


def get_predictor() -> SpamPredictor:
    predictor = runtime.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {runtime.get('error') or 'not loaded'}",
        )
    return predictor


def verify_api_key(request: Request) -> None:
    expected = os.environ.get("SPAMRADAR_API_KEY")
    if expected and request.headers.get("X-API-Key") != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/", include_in_schema=False)
async def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(index_path)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = WEB_DIR / "favicon.svg"
    if path.exists():
        return FileResponse(path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="favicon not found")


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(request: PredictRequest):
    predictor = get_predictor()
    try:
        return predictor.predict(request.text)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict_batch", response_model=BatchResponse, dependencies=[Depends(verify_api_key)])
async def predict_batch(request: BatchPredictRequest):
    predictor = get_predictor()
    try:
        predictions = predictor.predict_batch(request.messages)
    except Exception as exc:
        log.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")
    summary = {
        "spam": sum(1 for p in predictions if p["label"] == "SPAM"),
        "suspicious": sum(1 for p in predictions if p["label"] == "SUSPICIOUS"),
        "normal": sum(1 for p in predictions if p["label"] == "NORMAL"),
        "mean_latency_ms": round(
            sum(p["latency_ms"] for p in predictions) / max(1, len(predictions)), 3
        ),
    }
    return {
        "count": len(predictions),
        "predictions": predictions,
        "summary": summary,
        "latency_ms": round(sum(p["latency_ms"] for p in predictions), 3),
    }


@app.get("/health")
async def health():
    predictor = runtime.get("predictor")
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "error": runtime.get("error"),
                "version": "1.0.0",
            },
        )
    return predictor.health()


@app.get("/stats")
async def stats():
    predictor = get_predictor()
    return predictor.stats()


@app.get("/examples")
async def examples():
    predictor = runtime.get("predictor")
    if predictor is not None:
        return {"examples": predictor.examples()}
    return {"examples": CONFIG.get("ui", {}).get("examples", [])}


@app.get("/languages")
async def languages():
    return {
        "languages": [
            "English", "Bengali", "Banglish", "Hindi", "Hinglish",
            "Spanish", "French", "Arabic", "mixed-script",
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=CONFIG.get("api", {}).get("host", "127.0.0.1"),
        port=int(CONFIG.get("api", {}).get("port", 8000)),
        reload=False,
    )
