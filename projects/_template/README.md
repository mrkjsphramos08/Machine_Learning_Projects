# Project Template — copy me to start a new project

This folder is the **scaffold** for every project in this repository. It is a
working project: the tests pass and the pipeline runs. You just point it at a
different dataset.

---

## 1. Create a new project

From the repo root (`MLOps/`):

```powershell
Copy-Item -Recurse projects\_template projects\02_kaggle_<dataset>
```

Then edit **`configs/config.yaml`** only:

```yaml
project_name: "02_kaggle_<dataset>"      # folder name
experiment_name: "02_kaggle_<dataset>"   # MUST be unique — one per project
run_name: "baseline"
```

Optionally rename `tests/test_pipeline.py` to `tests/test_<something>.py` for a
friendlier test report. It is not required for uniqueness.

---

## 2. Get your data

### Kaggle datasets (recommended)

```powershell
pip install -r requirements-dev.txt          # once, from the repo root
python -c "import kagglehub; print(kagglehub.dataset_download('owner/dataset-name'))"
```

`kagglehub` needs no API token for public datasets. Copy the printed path's CSV
into `data/raw/` and keep that filename in `configs/config.yaml` under
`paths.raw_data`.

### Then version it with DVC (never commit data to Git)

```powershell
cd projects\02_kaggle_<dataset>
dvc add data/raw/<your_file>.csv     # writes <your_file>.csv.dvc
```

Commit the `.dvc` pointer file (not the CSV) from the repo root:

```powershell
cd ..\..
git add projects/02_kaggle_<dataset>/data/raw/<your_file>.csv.dvc
```

> `dvc add` must be run from inside the project folder when the file is there,
> because the `.dvc` pointer is written next to the data file.

---

## 3. Set up the ML problem in the config

```yaml
dataset:
  target_column: "price"     # <-- your label column
  test_size: 0.2
  random_state: 42
model:
  type: "RandomForestRegressor"
  n_estimators: 100
  max_depth: 6
```

Regression? You are done — `src/train.py` already logs RMSE/MAE/R². For
**classification**, extend `evaluate_model()` and `build_model()` in
`src/train.py` (log `accuracy`/`f1`/`roc_auc` instead) and update the metric
assertions in the test file.

---

## 4. Explore before you model

Put `01_eda.ipynb` in `notebooks/`. Learn the data first: shape, dtypes, missing
values, target distribution, obvious leakage. See `notebooks/README.md` for the
notebook → `src/` graduation path.

---

## 5. Run the pipeline

```powershell
cd projects\02_kaggle_<dataset>
python -m src.data       # or: dvc repro from here
python -m src.train
pytest
```

From the repo root you can run everything at once:

```powershell
dvc repro -P
mlflow ui --port 5000    # all projects' runs in one UI
```

---

## Rules this template already follows

| Rule (from `.agents/rules/mlops_standards.md`) | How it is honoured |
| :--- | :--- |
| Config decoupled from code | everything tunable is in `configs/config.yaml` |
| Explicit reproducibility | `random_state` propagated for split and model |
| Modular structure | `src/data.py` = ingestion, `src/train.py` = fit/evaluate |
| Type annotations | all function signatures are annotated |
| Named MLflow experiments | `experiment_name` from config, unique per project |
| Comprehensive logging | params + metrics + model artifact per run |
| No heavy binaries in Git | `data/` and `models/` are gitignored, DVC-tracked |
| Smoke + validation tests | `tests/test_pipeline.py`, hermetic via `tmp_path` |

## Things that are deliberately NOT in the template

Docker, FastAPI serving and the MLflow Model Registry. Those belong to the
**graduation** step: build them once, for the one project you actually want to
ship, following `projects/01_diabetes_regression/ROADMAP.md`. Adding them to
every experiment is how people end up doing MLOps instead of ML.
