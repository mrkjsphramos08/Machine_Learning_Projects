---
name: mlflow-model-registry
description: Procedures for logging runs, registering models in the MLflow Model Registry, promoting versions with movable aliases (registry stages are deprecated), and loading models for inference.
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

    # Log the model artifact AND register it under a unified model name.
    # MLflow 3 renamed `artifact_path` to `name`.
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="<project_model_name>",   # e.g. "diabetes_regression"
    )
```

---

## 2. Managing the lifecycle with Aliases (stages are deprecated)

⚠️ **Verified against MLflow 3.16.0:** `transition_model_version_stage` still
executes but emits
`FutureWarning: ... is deprecated since 2.9.0. Model registry stages will be
removed in a future major release.`

Use **aliases** instead. They are movable named pointers to a version, and they
are what `models:/<name>@<alias>` resolves against:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Point the "challenger" alias at version 2 (the candidate under evaluation)
client.set_registered_model_alias(
    name="diabetes_regression", alias="challenger", version=2
)

# After the gate passes, promote it: move "champion" onto the same version
client.set_model_version_tag(
    "diabetes_regression", 2, "promoted_reason", "rmse 51.2 -> 48.9 on frozen holdout"
)
client.set_registered_model_alias(
    name="diabetes_regression", alias="champion", version=2
)
client.delete_registered_model_alias(name="diabetes_regression", alias="challenger")
```

Alias vocabulary used in this repo: `champion` (serving), `challenger`
(under evaluation), `latest` (diagnostics only — never serve from it).

**Rollback is one call** — this is the main reason to prefer aliases:

```python
client.set_registered_model_alias(name="diabetes_regression", alias="champion", version=1)
```

For the full promotion policy and the evaluation procedure, see the
`model-promotion-and-aliases` skill.

---

## 3. Loading Models from the Registry for Inference

Load production models dynamically without knowing the file path or exact run ID:

```python
import mlflow.pyfunc
import pandas as pd

# Load the current champion by ALIAS (registry stages are deprecated)
model_uri = "models:/<project_model_name>@champion"
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
# Launch the tracking UI + Model Registry (run from the REPO ROOT)
mlflow ui --port 5000

# MLflow 3.x already defaults to sqlite:///mlflow.db resolved from the CURRENT
# directory, and that default enables the registry. To be explicit:
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

Because the default is relative to the current directory, `mlflow ui` only shows
your runs when it is started from the repo root — see §5.

---

## 5. How this monorepo uses MLflow

| Concern | Convention here |
| :--- | :--- |
| Tracking store | one shared `mlflow.db` at the repo root, for every project |
| Pinning | `configure_tracking()` in each project's `src/train.py` sets the URI explicitly; without it MLflow creates a stray `mlflow.db` in whatever folder you ran from |
| Artifact root | `ensure_experiment()` pins it to `mlruns/<experiment_name>/`; MLflow otherwise derives it from the current directory too |
| Experiment naming | `experiment_name` in `configs/config.yaml`, **unique per project** |
| Model naming | one registered model name per project, e.g. `"diabetes_regression"` |
| Reproducibility tag | runs normally carry `mlflow.source.git.commit`; log it explicitly (see `reproducibility-and-environments`) because it is not always captured |

```powershell
# verify the store resolves to the repo-level database
.\.venv\Scripts\python.exe -c "import mlflow; print(mlflow.get_tracking_uri())"
# -> sqlite:///C:/.../MLOps/mlflow.db
```

One project's `train.py` therefore looks like:

```python
tracking_uri = configure_tracking(config)      # repo-level mlflow.db
ensure_experiment(experiment_name, config)     # mlruns/<experiment_name>/
with mlflow.start_run(run_name=run_name):
    ...
```
