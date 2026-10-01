# Project 03: Titanic Passenger Survival Classification

A binary classification project predicting whether a passenger survived the Titanic sinking (`1` = Survived, `0` = Died).

---

## 1. Problem & Dataset Overview
- **Dataset**: Kaggle Titanic Dataset (`data/raw/titanic.csv`), tracked via DVC.
- **Rows & Columns**: 891 rows × 12 features.
- **Target**: `Survived` (Binary: `0` or `1`).
- **Core Challenges**:
  - Missing values in `Age` (~20%), `Cabin` (~77%), and `Embarked` (2 rows).
  - Categorical feature encoding (`Sex`, `Embarked`, `Pclass`).
  - Text feature extraction (`Name` titles like Mr., Mrs., Miss, Master).
  - Evaluation trade-offs: Accuracy vs. Precision, Recall, F1, and ROC-AUC.

---

## 2. Pipeline Architecture
```
data/raw/titanic.csv (DVC Tracked)
       │
       ▼
src/data.py (Clean, Impute, Encode, Split)
       │
       ├── data/processed/train.parquet
       └── data/processed/test.parquet
       │
       ▼
src/train.py (Fit Classifier, Log to MLflow, Evaluate)
       │
       ▼
models/model.pkl (Artifact)
```

---

## 3. Quickstart

### Step 1: Run Preprocessing
```powershell
cd projects\03_titanic_classification
python -m src.data
```

### Step 2: Train Model
```powershell
python -m src.train
```

### Step 3: Run Tests
```powershell
pytest
```
