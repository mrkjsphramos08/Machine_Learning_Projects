# MLOps Lab — one repo, many ML projects

A personal lab for learning MLOps by actually doing Machine Learning. One
repository, one virtual environment, one experiment-tracking database, and a
folder per project.

---

## 🧠 How this repo is organised

There are two tiers, and keeping them separate is the whole point:

| Tier | Where | What lives there |
| :--- | :--- | :--- |
| **The lab bench** — where you do ML | `projects/<name>/` | notebooks, data, configs, training scripts, tests. Self-contained. |
| **The platform** — what you set up once | repo root | virtualenv, `requirements*.txt`, `mlflow.db`, `.dvc` cache, `pytest.ini`, `.agents/` rules and skills |

You spend most of your time in `projects/`. The root only changes when you add a
library or fix tooling.

```
MLOps/
├── .agents/                        # rules (.agents/rules/mlops_standards.md) + skills
├── .dvc/                           # shared DVC cache (data is stored here, not in Git)
├── .venv/                          # ONE virtualenv shared by all projects
├── projects/
│   ├── _template/                  # copy this folder to start a new project
│   │   ├── configs/config.yaml     # every tunable value lives here
│   │   ├── src/{paths,data,train}.py
│   │   ├── tests/test_pipeline.py  # hermetic smoke + data-validation tests
│   │   ├── notebooks/README.md
│   │   ├── conftest.py             # makes `import src` work under pytest
│   │   ├── dvc.yaml                # prepare → train pipeline stages
│   │   └── README.md               # step-by-step new-project guide
│   └── 01_diabetes_regression/     # the reference project (+ ROADMAP.md, 4 milestones)
├── mlflow.db                       # shared SQLite tracking store + Model Registry
├── mlruns/                         # MLflow artifact store (gitignored)
├── pytest.ini                      # test configuration for the whole repo
├── requirements.txt                # shared runtime dependencies
├── requirements-dev.txt            # notebooks, plotting, Kaggle downloads
└── README.md
```

---

## 🚀 Quickstart

```powershell
# 1. Activate the shared virtualenv (Python 3.11)
.\.venv\Scripts\Activate.ps1
# If PowerShell blocks scripts:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 2. Install dependencies once for the whole repo
pip install -r requirements.txt          # to RUN pipelines
pip install -r requirements-dev.txt      # to do EDA / download Kaggle data

# 3. Work in a project
cd projects\01_diabetes_regression
python -m src.data       # build data/processed/*.parquet
python -m src.train      # train, evaluate, log to MLflow, write models/model.pkl
pytest                   # this project's tests only

# 4. Explore every run in one UI (run from the REPO ROOT)
cd ..\..
mlflow ui --port 5000    # http://localhost:5000
```

### Reproducible pipelines with DVC

```powershell
dvc repro -P                                       # reproduce ALL projects' pipelines
dvc repro projects/01_diabetes_regression/dvc.yaml # reproduce ONE project
dvc stage list --all                                     # cached vs stale stage status
dvc dag projects/01_diabetes_regression/dvc.yaml                                            # show the pipeline graph
```

DVC runs each project's stages with the **project folder** as the working
directory, which is why stage commands are plain `python -m src.data`.

---

## ➕ Starting a new project (including a Kaggle one)

```powershell
Copy-Item -Recurse projects\_template projects\02_kaggle_<dataset>
```

1. Edit `projects/02_kaggle_<dataset>/configs/config.yaml` — set `project_name`,
   `experiment_name` (**must be unique per project**) and `dataset.target_column`.
2. Get the data. `kagglehub` needs no API token for public datasets:
   ```powershell
   python -c "import kagglehub; print(kagglehub.dataset_download('owner/dataset-name'))"
   ```
   Copy the file into the project's `data/raw/`, then version it:
   ```powershell
   cd projects\02_kaggle_<dataset>
   dvc add data/raw/<your_file>.csv      # commit the generated .csv.dvc to Git
   ```
3. Explore it in `notebooks/01_eda.ipynb` **before** training anything.
4. `python -m src.data`, `python -m src.train`, `pytest`.

`projects/_template/README.md` has the long-form version of this checklist.

---

## 🔗 Shared services at the repo root (and why)

| Thing | Why it is shared, not per-project |
| :--- | :--- |
| **`.venv`** | One environment to maintain. If two projects ever need conflicting versions, that is the signal to split one out into its own repo. |
| **`mlflow.db`** | Every project's runs land in one UI so you can compare runs **across** datasets. |
| **`mlruns/<experiment_name>/`** | Model artifacts, one folder per experiment. `src/train.py` pins the tracking store *and* the artifact root to the repo root — MLflow 3.x otherwise derives both from the *current directory* and scatters stray databases and artifact folders into whatever folder you ran from. |
| **`.dvc/` cache** | Data is stored once and deduplicated, even if two projects use the same dataset. |
| **`pytest.ini`** | `--import-mode=importlib` lets every project ship its own `tests/test_pipeline.py` without module-name clashes. |
| **`.agents/` rules + skills** | The standards and the DVC / MLflow / FastAPI / Docker procedures apply to every project automatically. |

`experiment_name` is the only thing that must never be duplicated between
projects — that is what keeps the shared tracking store readable.

---

## 🧭 The promotion path

A project moves through these stages, and only the last one justifies Docker:

```
notebooks/  →  src/*.py  →  dvc.yaml stages  →  MLflow Model Registry
                                                    →  FastAPI  →  Docker
   exploration    reusable      reproducible        lifecycle     serving   packaging
                  functions     pipeline            management
```

`projects/01_diabetes_regression/ROADMAP.md` walks the whole path in four
milestones (DVC → MLflow registry → FastAPI → Docker). Do those steps **once**,
on one project. For everything else, stop at `dvc.yaml` + `pytest` and spend the
rest of your time on data quality, validation splits and the right metric.

---

## 📐 Conventions

Full rules: `.agents/rules/mlops_standards.md`. The essentials:

- Never hardcode hyperparameters, paths, split ratios or seeds — put them in
  `configs/config.yaml`.
- Always set `random_state` explicitly and log it.
- Log **all** runs to MLflow, including bad ones. Three runs you can compare beat
  one run you remember.
- Raw and processed data, model binaries and `mlruns/` never go into Git — DVC
  and MLflow handle them.
- Every pipeline component keeps a smoke test in `tests/`.

---

## 🧪 Verify the setup works

```powershell
pytest                                  # from the repo root: all projects
dvc repro -P                            # both pipeline stages still reproduce
dvc stage list --all                    # stage status across all pipelines
```
