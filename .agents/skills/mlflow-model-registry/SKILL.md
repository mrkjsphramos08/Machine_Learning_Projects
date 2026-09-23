---
name: mlflow-model-registry
description: Procedures for logging runs, registering models into the MLflow Model Registry, promoting models across lifecycle stages, and programmatically loading models for inference.
---

# MLflow Experiment Tracking & Model Registry Skill

The MLflow Model Registry provides a centralized model store, set of APIs, and UI to collaboratively manage the full lifecycle of an MLflow Model.

---

## 1. Registering a Model During Training

When logging the model inside `src/train.py`, pass `registered_model_name` to register it automatically:

```python
import mlflow.sklearn

with mlflow.start_run(run_name="rf_experiment_v1"):
    # ... training & metric logging ...

    # Log model artifact AND register under a unified model name
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="WineQualityRegressor"
    )
```

---

## 2. Managing Model Lifecycle & Stages Programmatically

Use the `MlflowClient` to transition models between lifecycle stages:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model version 1 to Staging
client.transition_model_version_stage(
    name="WineQualityRegressor",
    version=1,
    stage="Staging",
    archive_existing_versions=False
)

# Transition champion model to Production (and archive previous production models)
client.transition_model_version_stage(
    name="WineQualityRegressor",
    version=2,
    stage="Production",
    archive_existing_versions=True
)
```

---

## 3. Loading Models from the Registry for Inference

Load production models dynamically without knowing the file path or exact run ID:

```python
import mlflow.pyfunc
import pandas as pd

# Load the latest Production model directly via URI
model_uri = "models:/WineQualityRegressor/Production"
production_model = mlflow.pyfunc.load_model(model_uri)

# Run inference
sample_input = pd.DataFrame([{
    "feature_1": 0.038,
    "feature_2": 0.050,
    # ...
}])
predictions = production_model.predict(sample_input)
print(f"Prediction: {predictions}")
```

---

## 4. Useful MLflow CLI Commands

```powershell
# Launch the MLflow tracking server and Model Registry UI
mlflow ui --port 5000

# Use SQLite backend (enables full Model Registry support locally)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```
