# MLOps Project & Engineering Standards

These rules define the engineering standards for developing and deploying machine learning pipelines in this repository.

---

## 1. Code & Pipeline Architecture

- **Decouple Configuration from Code**: Never hardcode model hyperparameters, dataset paths, train/test split ratios, or random seeds in Python scripts. Always store these in `configs/*.yaml` and load them at runtime.
- **Reproducibility**: Always explicitly define, log, and propagate `random_state` across data splitting, preprocessing, and model initialization.
- **Modularity**: Structure logic into dedicated modules inside `src/`:
  - `src/data.py` for data ingestion, cleaning, and transformation.
  - `src/train.py` for model fitting, evaluation, and serialization.
  - `src/serve.py` or `src/api/` for inference serving.
- **Type Annotations**: Include type hints (`typing` / built-in types) on all function inputs and return signatures to ensure maintainability and testability.

---

## 2. Experiment Tracking & Model Packaging (MLflow)

- **Named Experiments**: Always organize runs under an explicit experiment name using `mlflow.set_experiment("<name>")`.
- **Comprehensive Logging**:
  - Log all configuration inputs using `mlflow.log_params(config_dict)` or `mlflow.log_param()`.
  - Log standard regression/classification metrics (`rmse`, `mae`, `r2`, `accuracy`, `f1`, etc.) with `mlflow.log_metric()`.
  - Save the model artifact using `mlflow.<flavor>.log_model()` so it is packaged with its dependencies and can be promoted to the Model Registry.
- **Run Context**: Always use the context manager `with mlflow.start_run():` to ensure runs close properly even on exceptions.

---

## 3. Data & Artifact Versioning (DVC)

- **Never Commit Heavy Binaries**: Never commit raw dataset files (`.csv`, `.parquet`, `.json`), serialized model weights (`.pkl`, `.onnx`), or the local `mlruns/` folder to Git.
- **DVC Tracking**: Use DVC (`dvc add <data_file>` or `dvc.yaml` pipeline stages) to track and version dataset snapshots alongside Git commit hashes.

---

## 4. Testing & Continuous Integration

- **Automated Smoke Tests**: Every pipeline component must have a corresponding test in `tests/` to verify it runs end-to-end without crashing.
- **Data Validation Tests**: Include unit tests that verify expected column names, non-empty dataframes, and data types before feeding data into models.
