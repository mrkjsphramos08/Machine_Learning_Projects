---
name: fastapi-model-serving
description: Best practices and templates for serving machine learning models as production REST APIs using FastAPI, Pydantic validation, health checks, and automated API tests.
---

# FastAPI Model Serving Skill

This skill outlines how to deploy machine learning models as robust, low-latency microservices with automatic validation, OpenAPI documentation, and test suites.

---

## 1. Directory Structure

```
src/
├── api/
│   ├── __init__.py
│   ├── app.py           # FastAPI application entrypoint
│   └── schemas.py       # Pydantic input and output schemas
```

---

## 2. Request & Response Schemas (`schemas.py`)

Using Pydantic ensures invalid requests (missing fields, wrong types) are rejected before reaching the model:

```python
from pydantic import BaseModel, Field
from typing import List

class ModelInput(BaseModel):
    features: List[float] = Field(..., description="List of feature values in expected order")

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str = "Production"
```

---

## 3. FastAPI Service Implementation (`app.py`)

```python
from fastapi import FastAPI, HTTPException
import numpy as np
import mlflow.pyfunc
from src.api.schemas import ModelInput, PredictionResponse

app = FastAPI(title="MLOps Model Serving API", version="1.0.0")

# Load model globally on startup
MODEL_URI = "models:/WineQualityRegressor/Production"
model = None

@app.on_event("startup")
def load_production_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception as e:
        print(f"Warning: Could not load production model from registry: {e}")

@app.get("/health", status_code=200)
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ModelInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    input_array = np.array([payload.features])
    pred = model.predict(input_array)
    return PredictionResponse(prediction=float(pred[0]))
```

---

## 4. Running and Testing the Server

```powershell
# Run the FastAPI server with auto-reload
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Interactive Swagger documentation: http://localhost:8000/docs
```

### Automated API Testing with Pytest:
```python
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```
