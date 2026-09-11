# MLOps Hands-On Starter Project

A minimal, production-grade starter repository designed for learning core MLOps concepts step-by-step: modular code structure, configuration management, experiment tracking, model artifact packaging, and automated smoke testing.

---

## 📁 Repository Structure

```
MLOps/
├── .gitignore              # Ignores virtualenv, heavy data, model binaries, and mlruns/
├── README.md               # Project guide and walkthrough
├── requirements.txt        # Pinned core libraries (scikit-learn, MLflow, PyYAML, pytest, DVC)
├── configs/                # Hyperparameters, paths, and pipeline configurations
│   └── train_config.yaml   # Separates configuration from code
├── data/                   # Data directory (tracked via .gitkeep, data files gitignored)
│   ├── raw/                # Immutable original source data
│   └── processed/          # Cleaned, transformed data ready for modeling
├── models/                 # Local export directory for exported artifacts (gitignored)
├── notebooks/              # Jupyter notebooks for exploratory data analysis (EDA)
│   └── README.md           # Best practices for transitioning from EDA to modular code
├── src/                    # Production-ready, modular Python package
│   ├── __init__.py
│   ├── data.py             # Data loading and preprocessing functions
│   └── train.py            # Training pipeline with MLflow tracking
└── tests/                  # Automated test suite
    ├── __init__.py
    └── test_train.py       # Smoke tests ensuring the pipeline runs without errors
```

---

## 🚀 Quickstart Guide

### 1. Activate the Virtual Environment
The virtual environment `.venv` has been created using Python 3.11.

In PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts scripts, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, or run `.\.venv\Scripts\activate.bat` in CMD).*

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Training Script
```powershell
python -m src.train
```

### 4. Run the Smoke Tests
```powershell
pytest
```

---

## 💡 The "Aha!" Moment: Experiment Tracking with MLflow

### What is the `mlruns/` folder?
When you run `python -m src.train`, you will notice a folder named `mlruns/` created automatically in your root directory.
- **What it is**: `mlruns/` is MLflow's default local file-based database. Every time an experiment runs, MLflow creates subfolders containing YAML files with your logged parameters, timestamped metric values, and serialized model binaries.
- **Why it's in `.gitignore`**: You should **never commit `mlruns/` to Git**. Tracking data and model binaries become very large quickly and are not source code. In production, this local folder is replaced by a remote tracking server (e.g., AWS S3 + PostgreSQL or Databricks).

### Try the Comparison Workflow:
1. **Run 1 (Baseline)**:
   Run the training script with default settings:
   ```powershell
   python -m src.train
   ```
2. **Run 2 (Experiment)**:
   Open `configs/train_config.yaml` and modify:
   ```yaml
   run_name: "experiment_deeper_trees"
   model:
     n_estimators: 200
     max_depth: 12
   ```
   Run the script again:
   ```powershell
   python -m src.train
   ```
3. **Open the MLflow UI**:
   Launch the web UI from your terminal:
   ```powershell
   mlflow ui --port 5000
   ```
4. **Compare**:
   Open [http://localhost:5000](http://localhost:5000) in your browser.
   - Click on the `wine_quality_prediction` experiment in the sidebar.
   - Select both runs and click **Compare**.
   - See side-by-side parameter changes, metric differences (RMSE, R2), and inspected model artifacts without having to manually record notes!
