# Project Context & Domain Vocabulary

## Monorepo Architecture
- **Platform Tier (Repo Root)**: Shared environment (`.venv`), dependencies (`requirements.txt`, `requirements-dev.txt`), shared tracking database (`mlflow.db`), shared artifact root (`mlruns/`), DVC configuration (`.dvc/`), testing root (`pytest.ini`), and automated agents (`.agents/`).
- **Project Tier (`projects/<name>/`)**: Independent, self-contained machine learning projects.
  - Structure: `configs/config.yaml`, `data/raw/` (immutable, DVC-tracked), `data/processed/`, `models/`, `notebooks/`, `src/{paths,data,train}.py`, `tests/`, `dvc.yaml`.
  - Scaffolding: Inherited from `projects/_template/`.

## Existing Projects
- `projects/01_diabetes_regression`: Reference implementation for regression using scikit-learn diabetes dataset (442 rows, 10 features, target progression, RMSE/R2).
- `projects/_template`: Reusable scaffold for new projects.

## Core Domain Vocabulary & Invariants
- **Data Ingestion & Versioning**: `data/raw/` holds immutable files tracked by DVC (`.dvc` files committed to Git; never raw binaries). Accompanied by provenance record in `data/raw/DATASET.md`.
- **Config-Driven Architecture**: Decoupled parameters in `configs/config.yaml` (`project_name`, `experiment_name`, `dataset.*`, `model.*`). No hardcoded paths or hyperparams.
- **MLflow Tracking & Registry**: Unique `experiment_name` per project in shared `mlflow.db`. Model promotion via aliases (`@champion`, `@challenger`), avoiding deprecated stages.
- **Reproducibility**: Explicit `random_state` throughout splits and model instantiation; git commit tagging.
- **Hermetic Testing**: Unit and pipeline smoke tests using `tmp_path` and isolated SQLite MLflow stores.
