---
name: fastapi-model-serving
description: Procedures and templates for serving a registered model as a REST API with FastAPI - lifespan-based model loading, Pydantic request/response validation, health checks, and API tests. Use when a project graduates to real-time serving. API tests require httpx2.
---

# FastAPI Model Serving Skill

This skill outlines how to deploy machine learning models as robust, low-latency microservices with automatic validation, OpenAPI documentation, and test suites.

---

## 1. Directory Structure

This repo is a monorepo: the service belongs to **one** graduated project, so the
code lives inside that project, not at the repo root.

```
projects/<name>/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py           # FastAPI application entrypoint
│   │   └── schemas.py       # Pydantic input and output schemas
│   ├── paths.py             # PROJECT_ROOT / REPO_ROOT / resolve()
│   ├── data.py
│   └── train.py
└── configs/config.yaml      # model.registered_name + serving.alias live here
```

Two consequences of the monorepo layout:

- Run the server **from the project folder** (`cd projects/<name>`), because the
  service reads `configs/config.yaml` and `src/` from the working directory.
- The shared `mlflow.db` and `.venv` stay at the repo root — `src/paths.py`
  already exposes `REPO_ROOT` for exactly this.

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
    model_uri: str = Field(..., description="Registry URI that produced this prediction")
    # Echoing the model URI makes a wrong-alias incident visible in the response.
```

---

## 3. FastAPI Service Implementation (`app.py`)

```python
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
import yaml

from fastapi import FastAPI, HTTPException
from src.api.schemas import ModelInput, PredictionResponse
from src.paths import REPO_ROOT

# Run from the PROJECT folder, so these relative paths resolve.
with open("configs/config.yaml", "r", encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh)

# Load by ALIAS, never by file path or run id, so promotions are picked up:
#   models:/<registered_name>@<alias>
MODEL_URI = (
    f"models:/{CONFIG['model']['registered_name']}"
    f"@{CONFIG.get('serving', {}).get('alias', 'champion')}"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE at startup, not per request.

    Note: `@app.on_event("startup")` is deprecated — the installed FastAPI
    0.141 source says "You should instead use the `lifespan` handlers".
    """
    mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
    try:
        app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"[+] Loaded {MODEL_URI}")
    except Exception as exc:                 # stay up, report unhealthy instead
        app.state.model = None
        print(f"[!] Could not load {MODEL_URI}: {exc}")
    yield
    app.state.model = None


app = FastAPI(title="Model Serving API", version="1.0.0", lifespan=lifespan)
```

Add the config keys this expects:

```yaml
model:
  registered_name: "diabetes_regression"
serving:
  alias: "champion"
```

@app.get("/health", status_code=200)
def health_check():
    return {
        "status": "healthy",
        "model_uri": MODEL_URI,
        "model_loaded": getattr(app.state, "model", None) is not None,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ModelInput):
    model = getattr(app.state, "model", None)   # set by the lifespan handler
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    input_array = np.array([payload.features])
    pred = model.predict(input_array)
    return PredictionResponse(prediction=float(pred[0]), model_uri=MODEL_URI)
```

---

## 4. Running and Testing the Server

```powershell
# Run from the PROJECT folder
cd projects\<name>
..\..\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Interactive Swagger documentation: http://localhost:8000/docs
```

### Automated API testing with pytest

⚠️ **Verified on this stack** (FastAPI 0.141.1 / Starlette 1.6.0):
`from fastapi.testclient import TestClient` raises
`RuntimeError: The starlette.testclient module requires the httpx2 package to be
installed.` — note **`httpx2`**, not `httpx`.

```powershell
pip install httpx2
```

```python
from fastapi.testclient import TestClient

from src.api.app import app


def test_health_reports_model_state():
    # The `with` block matters: it runs the lifespan startup/shutdown.
    # A bare TestClient(app) does NOT start the lifespan, so app.state.model
    # would never be set and the test could pass against a broken app.
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "model_uri" in body


def test_predict_rejects_bad_payload():
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": "not-a-list"})

    assert response.status_code == 422      # Pydantic validation, not a 500
```

Also test the **503 path**: when the model failed to load, the API must say so
rather than return a meaningless number. And test the happy path with a payload
built from `configs/config.yaml`'s feature list, so a column rename breaks a test
instead of production.
