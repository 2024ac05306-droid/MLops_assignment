import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field


MODEL_PATH = os.getenv("MODEL_PATH", "models/best_pipeline.pkl")

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_logger = logging.getLogger("api_requests")
api_logger.setLevel(logging.INFO)
api_logger.propagate = False
api_logger.handlers.clear()

request_log_formatter = logging.Formatter("%(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(request_log_formatter)
file_handler = logging.FileHandler(LOG_DIR / "api_requests.log")
file_handler.setFormatter(request_log_formatter)
api_logger.addHandler(console_handler)
api_logger.addHandler(file_handler)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "api_requests_in_progress",
    "Number of API requests currently being processed",
)
PREDICTION_COUNT = Counter(
    "api_predictions_total",
    "Total number of prediction requests completed successfully",
)

app = FastAPI(
    title="Heart Disease Model Serving API",
    version="1.0.0",
    description="Serves the trained heart disease classification pipeline.",
)


@app.middleware("http")
async def log_requests_and_collect_metrics(
    request: Request, call_next: Callable
) -> Response:
    start_time = time.perf_counter()
    method = request.method
    endpoint = request.url.path
    status_code = 500

    REQUESTS_IN_PROGRESS.inc()
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_seconds = time.perf_counter() - start_time
        REQUESTS_IN_PROGRESS.dec()
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration_seconds)

        api_logger.info(
            json.dumps(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "method": method,
                    "path": endpoint,
                    "status_code": status_code,
                    "duration_ms": round(duration_seconds * 1000, 2),
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                }
            )
        )


class HeartDiseaseInput(BaseModel):
    age: float = Field(..., example=63)
    sex: float = Field(..., example=1)
    cp: float = Field(..., example=1)
    trestbps: float = Field(..., example=145)
    chol: float = Field(..., example=233)
    fbs: float = Field(..., example=1)
    restecg: float = Field(..., example=2)
    thalach: float = Field(..., example=150)
    exang: float = Field(..., example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: float = Field(..., example=3)
    ca: float = Field(..., example=0)
    thal: float = Field(..., example=6)


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: dict[str, float]
    model_path: str


@lru_cache(maxsize=1)
def load_serving_model():
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file}. Run preprocessing and training first."
        )

    logger.info("Loading model from %s", model_file)
    return joblib.load(model_file)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartDiseaseInput):
    try:
        model = load_serving_model()
        input_data = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)

        prediction = int(model.predict(input_data)[0])

        probabilities: dict[str, float] = {}
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            probability_values = model.predict_proba(input_data)[0]
            classes = getattr(model, "classes_", range(len(probability_values)))
            probabilities = {
                str(int(label)): round(float(probability), 6)
                for label, probability in zip(classes, probability_values)
            }
            confidence = round(float(max(probability_values)), 6)

        logger.info("Prediction=%s confidence=%s", prediction, confidence)
        PREDICTION_COUNT.inc()
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_path": MODEL_PATH,
        }
    except FileNotFoundError as exc:
        logger.exception("Model loading failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
