---
name: new-project-scaffolding
description: Step-by-step procedure for creating a new ML project from projects/_template - naming, config wiring, dataset placement, DVC tracking, experiment naming, and the pre-flight checks to run before training.
---

# New Project Scaffolding Skill

Use this when starting any new project in this repo (a Kaggle dataset, a new
problem, a fresh idea). The goal is that every project looks identical, so the
only thing that changes between projects is the data and the model.

---

## 1. Name the project

Convention: `NN_<source>_<problem>`, lowercase with underscores.

| Good | Why |
| :--- | :--- |
| `02_kaggle_titanic_classification` | source and task are obvious from the folder |
| `03_uci_wine_quality` | |
| `04_energy_demand_forecast` | |

Avoid spaces and uppercase (they leak into experiment names and paths).

---

## 2. Copy the template

```powershell
cd <repo root>
Copy-Item -Recurse projects\_template projects\02_kaggle_titanic_classification
```

Then remove the template's README section you do not need and write a short
project `README.md`: dataset, source, rows/features, target, task, primary metric.

---

## 3. Wire the config (the only file you must edit to start)

`projects/<name>/configs/config.yaml`:

```yaml
project_name: "02_kaggle_titanic_classification"
experiment_name: "02_kaggle_titanic_classification"   # MUST be unique in mlflow.db
run_name: "baseline"

paths:
  raw_data: "data/raw/train.csv"        # whatever you downloaded
  train_data: "data/processed/train.parquet"
  test_data: "data/processed/test.parquet"
  model: "models/model.pkl"

dataset:
  target_column: "Survived"             # your label column
  test_size: 0.2
  random_state: 42

model:
  type: "RandomForestRegressor"         # change to a classifier for classification
  n_estimators: 100
  max_depth: 6
  random_state: 42
```

Everything else in the project reads from here — do not hardcode paths or
hyperparameters in Python (`.agents/rules/mlops_standards.md §1`).

> **Classification?** `build_model()` in `src/train.py` raises on anything other
> than `RandomForestRegressor`, and `evaluate_model()` logs RMSE/MAE/R2. Swap in
> `RandomForestClassifier` and log `accuracy`/`f1`/`roc_auc` — see the
> `metrics-and-evaluation` skill.

---

## 4. Place and version the dataset

See the `dataset-acquisition` skill for the download itself, then:

```powershell
cd projects\02_kaggle_titanic_classification
dvc add data/raw/train.csv          # writes data/raw/train.csv.dvc
cd ..\..
git add projects/02_kaggle_titanic_classification/data/raw/train.csv.dvc
```

`dvc add` must run from inside the project folder — the `.dvc` pointer is always
written next to the data file. Commit the `.dvc` file, never the CSV.

---

## 5. Look at the data before modelling

Create `notebooks/01_eda.ipynb` (see `exploratory-data-analysis`). You are looking
for: the target distribution, missing values, dtypes, obvious leakage, and
whether the split protocol you planned actually fits the data.

---

## 6. Run the pipeline

```powershell
cd projects\02_kaggle_titanic_classification
python -m src.data          # prepare stage
python -m src.train         # train stage
pytest                      # this project's tests only
```

Or from the repo root, `dvc repro -P` to reproduce every project.

---

## 7. Definition of done for a new project

- [ ] Every folder name, `project_name` and `experiment_name` agree.
- [ ] Raw data is DVC-tracked and the `.dvc` file is committed.
- [ ] `configs/config.yaml` holds every tunable; no literals in `src/`.
- [ ] `notebooks/01_eda.ipynb` exists and is committed.
- [ ] `python -m src.data` and `python -m src.train` run clean.
- [ ] `pytest` passes (6 tests out of the box).
- [ ] At least one MLflow run is visible under the new experiment name.
- [ ] The project `README.md` states the dataset, task and primary metric.

---

## 8. Pitfalls

| Symptom | Cause |
| :--- | :--- |
| Runs of two projects mixed together | `experiment_name` copied from the template |
| `mlflow.db` appears inside the project folder | MLflow was used without `configure_tracking()`; the repo pins it deliberately |
| `FileNotFoundError` on data | Running from the repo root instead of the project folder, or a path in code instead of `src/paths.py` |
| `dvc repro -P` fails on `_template` | The scaffold is excluded via `.dvcignore`; do not edit that rule |
| Two `tests/test_pipeline.py` clash | Impossible by design (`--import-mode=importlib` in `pytest.ini`) |
