"""FastAPI inference service for Diabetes Progression Regression.

Serves predictions using the model registered in MLflow under the `@champion` alias.
"""

from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from src.api.schemas import ModelInput, PredictionResponse
from src.paths import PROJECT_ROOT, REPO_ROOT

FEATURE_NAMES = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]

# Resolve configuration relative to the project root
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh)

REGISTERED_NAME = CONFIG.get("model", {}).get("registered_name", "DiabetesRegressor")
ALIAS = CONFIG.get("serving", {}).get("alias", "champion")
MODEL_URI = f"models:/{REGISTERED_NAME}@{ALIAS}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: load the champion model once into memory on startup."""
    tracking_uri = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    print(f"[*] Connecting to MLflow tracking store: {tracking_uri}")
    print(f"[*] Loading champion model: {MODEL_URI}")

    try:
        app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"[+] Successfully loaded {MODEL_URI}")
    except Exception as exc:
        app.state.model = None
        print(f"[!] Warning: Could not load {MODEL_URI}: {exc}")

    yield
    app.state.model = None


app = FastAPI(
    title="Diabetes Progression Prediction API",
    description="Real-time inference microservice serving the MLflow Champion model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root requests to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", status_code=200)
def health_check():
    """Health check endpoint: reports service status and model readiness."""
    is_loaded = getattr(app.state, "model", None) is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_uri": MODEL_URI,
        "model_loaded": is_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ModelInput):
    """Predict diabetes disease progression one year after baseline."""
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Champion model '{MODEL_URI}' is not loaded. Check MLflow registry.",
        )

    # Convert features to a DataFrame matching the model's logged signature schema
    input_df = pd.DataFrame([payload.features], columns=FEATURE_NAMES)

    try:
        prediction = model.predict(input_df)
        pred_value = float(prediction[0])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Inference error: {exc}",
        ) from exc

    return PredictionResponse(prediction=pred_value, model_uri=MODEL_URI)
