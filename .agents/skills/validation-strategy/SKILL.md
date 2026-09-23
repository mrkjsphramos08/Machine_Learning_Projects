---
name: validation-strategy
description: How to choose and implement the train/validation split protocol - holdout vs KFold, Stratified, Group and TimeSeries splits, leakage through splitting, and keeping the protocol reproducible inside a project.
---

# Validation Strategy Skill

Use this after EDA, before modelling. Changing the split changes the score more
than most hyperparameter changes — **the split protocol is part of the model**.

---

## 1. Choose by data structure, not by habit

| Data structure | Splitter | Why |
| :--- | :--- | :--- |
| Independent rows, balanced target | `train_test_split`, `KFold` | default case |
| Imbalanced classification | `stratified` / `StratifiedKFold` | preserves class ratio in every fold |
| Repeated entities (patient, customer, device) | `GroupKFold` / `GroupShuffleSplit` | same entity must never span the boundary |
| Time-ordered | `TimeSeriesSplit` (with a gap) | you cannot train on the future |
| Spatial / blocked (sensors, regions) | blocked or grouped split | nearby rows are correlated |
| Tiny dataset (< ~500 rows) | `LeaveOneOut` or repeated KFold | the holdout is too noisy to trust |

---

## 2. The protocol ladder used in this repo

1. **Iterate** on a single holdout split (fast, deterministic, seed-fixed).
2. **Select** between candidates with cross-validation on the training portion
   (`baseline-and-model-selection`).
3. **Report** once, on the frozen holdout, at the end.

Never swap step 3 for step 1: the test set is a report, not a tuning knob.

---

## 3. Implementation

**Holdout (what the template already does)**

```python
train_df, test_df = train_test_split(
    df,
    test_size=cfg["dataset"]["test_size"],
    random_state=cfg["dataset"]["random_state"],
    stratify=df[y] if is_classification else None,
)
```

**Stratified K-fold**

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Grouped**

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)
scores = cross_validate(model, X, y, cv=cv, groups=df["patient_id"], scoring="r2")
```

**Time series (note the gap)**

```python
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5, gap=1)  # gap prevents boundary leakage
```

**Tiny data** — repeated KFold with several seeds, report mean ± std, and do not
brag about a single run.

---

## 4. Leakage that enters *through* the split

| Leak | Detection | Fix |
| :--- | :--- | :--- |
| Duplicate rows on both sides | join train/test on all features | deduplicate before splitting |
| Same entity in both sides | `set(train[id]) & set(test[id])` | `GroupKFold` |
| Future rows used to predict the past | check date ranges per split | `TimeSeriesSplit` + gap |
| Preprocessing fitted before splitting | grep for `fit_transform` outside a Pipeline | `feature-engineering` skill |
| Oversampling before splitting | SMOTE applied to the full frame | apply inside CV folds only |

A single leaked row can move RMSE more than an hour of tuning.

---

## 5. How this plugs into the repo

- The split happens once, in the `prepare` stage (`src/data.py`), driven by
  `configs/config.yaml`:

```yaml
dataset:
  target_column: "target"
  test_size: 0.2
  random_state: 42
  # optional, for structured splits:
  # group_column: "patient_id"
  # time_column: "date"
```

- `random_state` must be fixed and logged — it is already logged as
  `data_random_state` in every MLflow run.
- Outputs are `data/processed/train.parquet` and `test.parquet`, so the exact
  split is reproducible from the DVC lock file.
- If you move to grouped or time splits, the holdout is no longer a random
  sample — record that in the project README, because it changes what the score
  means.

---

## 6. Sizing guidance

| Rows | Holdout | CV |
| :--- | :--- | :--- |
| > 100k | 20% is plenty | 3–5 folds |
| 10k–100k | 20% | 5 folds |
| 1k–10k | 20–30% | 5–10 folds |
| < 1k | unreliable | repeated KFold / LOO, report variance |

---

## 7. Checklist

- [ ] Splitter matches the data structure (not just the default).
- [ ] `random_state` fixed, logged, and shared between split and model.
- [ ] Test set untouched until the final report.
- [ ] Entity/time leakage checks from §4 actually run.
- [ ] The chosen protocol is written down in the project README.
