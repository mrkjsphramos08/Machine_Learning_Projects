---
name: skills-index
description: Directory of every available skill in this repo, grouped by machine-learning workflow stage, with the trigger for invoking each one.
---

# Skills Index

Skills are **procedural, repo-anchored playbooks**. Each folder contains a
`SKILL.md` whose `description:` frontmatter states *when* to invoke it. Invoke a
skill when you are about to do the thing it describes — not to learn the theory.

```
DISCOVER → ACQUIRE → EXPLORE → PREPARE → MODEL → EVALUATE → PROMOTE → SERVE → MONITOR
```

---

## 1. Project setup & data

| Skill | Use when |
| :--- | :--- |
| `new-project-scaffolding` | Starting a new project (copying `projects/_template`), naming the experiment, wiring config paths |
| `dataset-acquisition` | Downloading a Kaggle/public dataset, recording provenance, versioning it with DVC |
| `exploratory-data-analysis` | Opening a new dataset in a notebook: shape, missingness, target distribution, leakage |
| `data-validation` | Turning dataset expectations into executable checks that run before training |

## 2. Features & validation design

| Skill | Use when |
| :--- | :--- |
| `feature-engineering` | Adding preprocessing to a project: imputation, encoding, scaling, datetime/text features |
| `validation-strategy` | Choosing or changing the split protocol (holdout, KFold, Group, TimeSeries) |

## 3. Modeling

| Skill | Use when |
| :--- | :--- |
| `baseline-and-model-selection` | Before trusting any score: dummy baselines, comparing candidates on one split |
| `hyperparameter-tuning` | Searching a hyperparameter space with CV, and logging the result to MLflow |
| `model-interpretability` | Explaining what drives predictions or diagnosing where the model fails |

## 4. Evaluation

| Skill | Use when |
| :--- | :--- |
| `metrics-and-evaluation` | Picking the right metric, reading a confusion matrix/residuals, handling class imbalance |
| `ml-testing-strategy` | Writing tests for ML code: smoke tests, invariants, hermetic fixtures |

## 5. Tracking, packaging & delivery

| Skill | Use when |
| :--- | :--- |
| `dvc-data-versioning` | Versioning data, building/reproducing `dvc.yaml` pipeline stages, DVC remotes |
| `mlflow-model-registry` | Logging runs, registering models, promoting versions with aliases |
| `model-promotion-and-aliases` | Champion/challenger comparison and release gating before promotion |
| `fastapi-model-serving` | Exposing a model as a REST API with validation and tests |
| `batch-scoring` | Scoring a file/table offline with a registry model |
| `docker-ml-packaging` | Containerizing training or the serving API |

## 6. Operations & reproducibility

| Skill | Use when |
| :--- | :--- |
| `reproducibility-and-environments` | Making a run reproducible, or reproducing an old run |
| `ml-cicd` | Automating lint, tests, pipeline and image builds in CI |
| `monitoring-and-drift` | After deployment: watching performance, drift, and retraining triggers |

---

## Ground rules that apply to every skill

1. **Project tier vs platform tier** — work happens in `projects/<name>/`; the
   venv, `requirements*.txt`, `mlflow.db`, `.dvc` cache, `pytest.ini` and
   `.agents/` live at the repo root. See `.agents/rules/mlops_standards.md §0`.
2. **Config, never hardcoding** — paths, splits and hyperparameters belong in
   `projects/<name>/configs/config.yaml`.
3. **Unique experiment name per project** — one shared `mlflow.db` holds every
   project's runs.
4. **Resolve paths from `src/paths.py`**, never from the current directory.
5. **Do not add Docker/FastAPI/registry wiring to every project.** Those are the
   graduation step for the one project you actually want to ship
   (`projects/01_diabetes_regression/ROADMAP.md`).

## Dependencies these skills assume

```powershell
# Tier 1 - EDA and dataset downloads
pip install -r requirements-dev.txt          # jupyter, ipykernel, seaborn, matplotlib, kagglehub

# Tier 2 - validation, tuning, interpretation
pip install pandera optuna shap

# Tier 3 - serving tests and monitoring
pip install httpx2 evidently
```

Core pipeline dependencies (`scikit-learn`, `pandas`, `numpy`, `pyarrow`,
`joblib`, `mlflow`, `dvc`, `pytest`, `pyyaml`) are in `requirements.txt`.
