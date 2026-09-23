# MLOps Project & Engineering Standards

These rules define the engineering standards for developing and deploying machine learning pipelines in this repository.

**Rules vs skills.** This file states the non-negotiables. The step-by-step *how* lives in `.agents/skills/` — see the index at `.agents/skills/README.md` for which skill to use at each stage of the ML workflow.

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
- **Pin MLflow to the repo root**: `configure_tracking()` points MLflow at the repo-level `mlflow.db`, and `ensure_experiment()` pins each experiment's artifact root to the repo-level `mlruns/`. Both values are otherwise derived from the current working directory, which silently scatters stray databases and artifact folders into project folders.
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
- **Provenance is mandatory**: every project's `data/raw/DATASET.md` records the source URL, retrieval date, licence and sha256 of the file. Without it the dataset cannot be audited or re-fetched.
- **Commit `dvc.lock` with the code**: the lock file ties a specific data hash to a specific code and config revision. Changing data or config without committing the lock file breaks the chain.
- **Configure a DVC remote** for any data you could not re-download by hand: the local `.dvc/cache` is a single point of failure. `dvc remote list` must not be empty once a project matters.

---

## 4. Testing & Continuous Integration

- **Automated Smoke Tests**: Every pipeline component must have a corresponding test in `tests/` to verify it runs end-to-end without crashing.
- **Data Validation Tests**: Include unit tests that verify expected column names, non-empty dataframes, and data types before feeding data into models.
- **Hermetic Tests**: Tests must not depend on downloaded data or on pipeline outputs. Build a tiny synthetic dataset in `tmp_path`, write a temporary config that points at it, and aim `mlflow.tracking_uri` at a temporary SQLite file so the shared `mlflow.db` stays clean.
- **Never assert a metric value**: asserting an exact RMSE/accuracy makes a test that breaks on every dependency bump. Assert invariants (determinism, output shape, output range) or a threshold relative to the baseline.
- **Keep the default suite fast**: mark anything slow with `@pytest.mark.slow` (registered in `pytest.ini`) and keep the unmarked suite under about 30 seconds.

---

## 5. Model Lifecycle & Release

- **Aliases, not stages**: `transition_model_version_stage` is deprecated (since MLflow 2.9). Promote with `set_registered_model_alias` and serve `models:/<name>@champion`. Never serve `@latest` — that is an untested version.
- **One registered model name per project**, read from `configs/config.yaml` (`model.registered_name`), so versions do not collide across projects in the shared registry.
- **Write the promotion gate before evaluating**: primary metric, the margin required (must exceed fold-to-fold std), the worst-slice limit, and the exact holdout used. A gate invented after seeing the numbers is not a gate.
- **Champion and challenger are scored on the identical frozen holdout** — never compare a new metric against an old one measured on different rows.
- **Log a signature and an input example** with every registered model (`infer_signature`, `input_example`) so serving can validate payloads instead of silently accepting nonsense.
- **A rollback target must exist**: keep the previous champion registered. Rollback is an alias move, not a retrain.
- **Batch scoring before real-time serving**: do not build an API for a model nobody consumes yet. Offline scoring is the cheaper path to real value.

---

## 6. Reproducibility & Operations

- **Log the git commit explicitly**: tag runs with `git_commit` (and note the data version via `dvc.lock`). MLflow does not reliably capture `mlflow.source.git.commit` for every run.
- **Two runs, identical metrics**: before trusting a result, confirm that re-running with the same seed reproduces it. If it does not, that is a bug, not noise.
- **Docker and FastAPI are graduation artifacts**, not project scaffolding: only one project at a time gets them (`projects/01_diabetes_regression/ROADMAP.md`). Adding them to every experiment is how ML work turns into YAML work.
- **Monitoring thresholds must have an owner**: every drift or performance alert names who looks and what they do. An unowned alert is noise.
- **The batch scoring job is the cheapest monitoring surface**: log row counts, prediction distributions and null rates on every scoring run.
- **Skills are the procedures** — when you need the how, read `.agents/skills/README.md` rather than improvising.
