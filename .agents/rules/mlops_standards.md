# MLOps Project & Engineering Standards

These rules define the engineering standards for developing and deploying machine learning pipelines in this repository.

---

## 0. Repository Layout (monorepo — two tiers)

This repository holds many independent ML projects. Keep the tiers separate:

- **Project tier — `projects/<name>/`** — where Machine Learning actually happens. Each project is self-contained and owns:
  - `configs/config.yaml` — every tunable (paths, split, hyperparameters, seeds)
  - `data/raw/` (immutable, DVC-tracked) and `data/processed/` (pipeline outputs)
  - `models/` (pipeline outputs), `notebooks/` (EDA and prototyping)
  - `src/{paths,data,train}.py`, `tests/`, `conftest.py`, `dvc.yaml`
  - Start a new project by copying `projects/_template/`.
- **Platform tier — the repo root** — set up once, shared by every project: `.venv`, `requirements.txt` / `requirements-dev.txt`, `mlflow.db`, the `.dvc/` cache, `pytest.ini`, `.agents/`.

Rules that follow from this:

- **One of each, never per project**: one virtualenv, one DVC repo, one MLflow tracking store. Use a unique `experiment_name` per project instead of separate stores.
- **Unique experiments**: `experiment_name` in `configs/config.yaml` must be unique per project so a shared `mlflow.db` stays readable.
- **Pin the tracking URI**: always point MLflow at the repo-level `mlflow.db` (see `configure_tracking()` in `src/train.py`). MLflow 3.x otherwise resolves `sqlite:///mlflow.db` relative to the current directory and silently creates stray databases.
- **Resolve paths from the file, not the CWD**: use `src/paths.py` (`PROJECT_ROOT` / `REPO_ROOT` / `resolve()`). Never assume the working directory.
- **No cross-project imports**: a project imports only from its own `src` package.

---

## 1. Code & Pipeline Architecture

- **Decouple Configuration from Code**: Never hardcode model hyperparameters, dataset paths, train/test split ratios, or random seeds in Python scripts. Always store these in `configs/*.yaml` and load them at runtime.
- **Reproducibility**: Always explicitly define, log, and propagate `random_state` across data splitting, preprocessing, and model initialization.
- **Modularity**: Structure logic into dedicated modules inside the project's `src/`:
  - `src/data.py` for data ingestion, cleaning, and transformation.
  - `src/train.py` for model fitting, evaluation, and serialization.
  - `src/paths.py` for path resolution (never depend on the current working directory).
  - `src/serve.py` or `src/api/` for inference serving (graduation step only).
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
- **Hermetic Tests**: Tests must not depend on downloaded data or on pipeline outputs. Build a tiny synthetic dataset in `tmp_path`, write a temporary config that points at it, and aim `mlflow.tracking_uri` at a temporary SQLite file so the shared `mlflow.db` stays clean.
