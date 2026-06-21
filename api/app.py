import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.model_loader import ModelWrapper, load_model
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.data.preprocessing import engineer_features

# ── Structured JSON Logging ──────────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """Outputs log records as single-line JSON for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def _configure_logging():
    """Configure structured JSON logging in production, human-readable in dev."""
    app_env = os.getenv("APP_ENV", "development")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if app_env == "production":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    root_logger.addHandler(handler)


_configure_logging()
logger = logging.getLogger(__name__)

_model: ModelWrapper = None
_start_time: float = time.time()

# ── Rate Limiter ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _start_time
    _start_time = time.time()
    logger.info("🚀 Starting Credit Risk API — loading model...")
    try:
        _model = load_model()
        logger.info(f"✅ Model loaded: {_model.name} v{_model.version}")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        # App still starts — health endpoint will report degraded
        _model = None
    yield
    logger.info("👋 Shutting down Credit Risk API.")


app = FastAPI(
    title="Credit Risk Prediction API",
    description=(
        "MLOps-grade REST API for predicting loan default probability "
        "using XGBoost trained on the Home Credit dataset. "
        "Tracks experiments with MLflow and monitors drift with Evidently AI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

# ── Serve Frontend Static Files ──────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _risk_label(prob: float) -> str:
    if prob < 0.20:
        return "LOW"
    elif prob < 0.50:
        return "MEDIUM"
    return "HIGH"


def _risk_score(prob: float) -> int:
    """Convert default probability to a 0–1000 credit risk score (higher = safer)."""
    return max(0, min(1000, int((1 - prob) * 1000)))


def _request_to_dataframe(req: PredictionRequest) -> pd.DataFrame:
    """Convert a single PredictionRequest to a 1-row DataFrame with feature engineering."""
    data = req.model_dump()
    df = pd.DataFrame([data])
    df = engineer_features(df)
    return df


def _predict_single(df: pd.DataFrame) -> float:
    """Run model inference on a 1-row DataFrame. Returns default probability."""
    try:
        proba = _model.predict_proba(df)
        return float(proba[0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root():
    """Serve the frontend dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"), status_code=200)


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health_check():
    """Liveness & readiness check."""
    if _model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version="N/A",
            uptime_seconds=round(time.time() - _start_time, 2),
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=_model.version,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("30/minute")
async def predict(payload: PredictionRequest, request: Request):
    """
    Predict loan default probability for a single application.

    Returns:
    - **default_probability**: float [0, 1]
    - **risk_label**: LOW / MEDIUM / HIGH
    - **risk_score**: 0–1000 (higher = safer)
    - **model_version**: MLflow model version

    Rate limit: 30 requests per minute per IP.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check /health for details.",
        )

    df = _request_to_dataframe(payload)
    prob = _predict_single(df)

    logger.info(
        f"Prediction: prob={prob:.4f} risk={_risk_label(prob)} | "
        f"credit={payload.AMT_CREDIT} income={payload.AMT_INCOME_TOTAL}"
    )

    return PredictionResponse(
        default_probability=round(prob, 6),
        risk_label=_risk_label(prob),
        risk_score=_risk_score(prob),
        model_version=_model.version,
        model_name=_model.name,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
@limiter.limit("10/minute")
async def predict_batch(payload: BatchPredictionRequest, request: Request):
    """
    Predict loan default probability for up to 100 applications in a single call.
    More efficient than repeated single-record calls.

    Rate limit: 10 requests per minute per IP.
    """
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    predictions = []
    probs = []

    for app_req in payload.applications:
        df = _request_to_dataframe(app_req)
        prob = _predict_single(df)
        probs.append(prob)
        predictions.append(
            PredictionResponse(
                default_probability=round(prob, 6),
                risk_label=_risk_label(prob),
                risk_score=_risk_score(prob),
                model_version=_model.version,
                model_name=_model.name,
            )
        )

    logger.info(f"Batch prediction: {len(predictions)} records | avg_prob={np.mean(probs):.4f}")

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        avg_default_probability=round(float(np.mean(probs)), 6),
    )


@app.get("/model/info", tags=["Ops"])
async def model_info():
    """Return metadata about the currently loaded model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    return {
        "model_name": _model.name,
        "model_version": _model.version,
        "registered_model": "credit-risk-model",
        "dataset": "Home Credit Default Risk (Kaggle)",
    }
