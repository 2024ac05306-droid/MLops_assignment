import logging
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Model Serving API",
    version="1.0.0",
    description="Serves the trained heart disease classification pipeline.",
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
