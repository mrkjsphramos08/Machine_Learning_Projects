---
name: dvc-data-versioning
description: Step-by-step procedures for initializing DVC, tracking dataset versions, configuring remote storage, and constructing reproducible multi-stage pipelines with dvc.yaml.
---

# DVC Data & Pipeline Versioning Skill

Data Version Control (DVC) enables Git-like versioning for datasets and machine learning models without bloating Git repositories.

---

## 1. Quickstart: Initialize DVC in the Workspace

```powershell
# Initialize DVC within the Git repository
dvc init

# View status
dvc status
```

Commit the generated internal `.dvc` files to Git:
```powershell
git add .dvc .dvcignore
git commit -m "chore: initialize DVC"
```

---

## 2. Versioning Datasets

### A. Tracking a Data File or Folder
When you have a dataset in `data/raw/data.csv`:
```powershell
# Track dataset with DVC (generates data/raw/data.csv.dvc and updates .gitignore)
dvc add data/raw/data.csv

# Commit the .dvc pointer file to Git
git add data/raw/data.csv.dvc data/raw/.gitignore
git commit -m "feat(data): track raw dataset version v1"
```

### B. Setting Up Remote Storage (Local Directory or S3/GCS)
```powershell
# Set up a local folder as a remote storage simulator (useful for learning)
dvc remote add -d local_storage /path/to/my_remote_storage

# Or configure an S3 bucket
# dvc remote add -d my_s3_remote s3://my-mlops-bucket/dvcstore

# Push tracked data to remote storage
dvc push

# Pull tracked data on another machine or branch
dvc pull
```

---

## 3. Building Multi-Stage Pipelines (`dvc.yaml`)

Define deterministic, cached pipeline stages in `dvc.yaml`:

```yaml
stages:
  prepare:
    cmd: python -m src.data
    deps:
      - src/data.py
      - configs/train_config.yaml
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet

  train:
    cmd: python -m src.train
    deps:
      - src/train.py
      - configs/train_config.yaml
      - data/processed/train.parquet
    outs:
      - models/model.pkl
```

### Running the Pipeline (monorepo note)

Each project keeps its own `dvc.yaml` inside `projects/<name>/`. From the repo
root, use `-P` (all pipelines) or pass the pipeline file explicitly:

```powershell
dvc repro -P                             # every pipeline in the repo
dvc repro projects/<name>/dvc.yaml       # one project
dvc stage list --all                     # stage status across all pipelines
dvc dag projects/<name>/dvc.yaml         # pipeline graph
```

From inside a project folder the plain forms work as usual:

```powershell
dvc repro     # stages re-run only when their dependencies changed
dvc dag
dvc status
```

`projects/_template` is excluded through `.dvcignore` because the scaffold has
no dataset of its own, so it is skipped by `dvc repro -P`.
