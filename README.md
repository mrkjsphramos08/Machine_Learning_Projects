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
