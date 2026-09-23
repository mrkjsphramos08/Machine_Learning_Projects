---
name: baseline-and-model-selection
description: Procedures for establishing baselines and comparing candidate models fairly - dummy baselines, one shared split, cross-validation with train scores, metric comparability, and bias-variance diagnosis before tuning.
---

# Baseline & Model Selection Skill

Use this **before** you believe any score and **before** you tune anything. The
purpose is to know whether your model is actually learning, and whether the
difference between two candidates is real or noise.

---

## 1. Always start with a dummy baseline

```python
from sklearn.dummy import DummyClassifier, DummyRegressor

DummyRegressor(strategy="mean")  # regression: predicts the mean
DummyClassifier(strategy="most_frequent")  # classification: majority class
```

Log it to MLflow exactly like a real model (`model_type="DummyRegressor"`). It is
the reference point for every later run. **A score you cannot beat with the mean
or the majority class is not a model.** Add a `LinearRegression` /
`LogisticRegression` run as the "simple but real" rung.

---

## 2. The honest comparison protocol

Five things must be identical across every candidate:

1. the same rows — same split, same `random_state`
2. the same preprocessing (`feature-engineering` skill)
3. the same folds — same CV splitter *and* seed
4. the same metric, computed the same way
5. the same evaluation rows — do not repeatedly peek at the test set and adapt

Each candidate gets its own MLflow run with `model_type` as a logged param, so the
UI comparison is apples-to-apples.

---

## 3. Compare with cross-validation, not a single split

```python
from sklearn.model_selection import KFold, cross_validate

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
    return_train_score=True,
)
```

- Prefer `cross_validate` over `cross_val_score`: `return_train_score=True` gives
  the bias/variance signal for free.
- Report **mean and standard deviation**. A model that is 0.01 better with 0.05
  std is not better.
- `scoring` strings are maximised by sklearn, hence `neg_`; flip the sign before
  reporting RMSE.

---

## 4. Read train vs CV score

| Train | CV | Diagnosis | Next move |
| :--- | :--- | :--- | :--- |
| much better | clearly worse | high variance (overfitting) | regularise, simplify, drop features, get more data |
| poor | poor | high bias (underfitting) | more capacity, better features, different family |
| good | close to train | healthy | tune lightly, then confirm on the held-out test set once |

This table is the reason not to tune first: tuning a high-variance model just
finds the most overfit setting.

---

## 5. Candidate ladder (cheapest first)

1. `Dummy*` — the floor
2. Linear / logistic regression — fast, interpretable, surprisingly strong
3. A single decision tree — the interpretable ceiling
4. `RandomForest*`
5. `HistGradientBoosting*` (shipped with sklearn — no extra dependency)
6. LightGBM / XGBoost / CatBoost — not installed yet; add only when 5 plateaus
7. Ensembling / stacking — only if the gain exceeds the noise band

Write the winner into `configs/config.yaml` as `model.type`, and record in the
project README: which model, which metric, which CV protocol, why.

---

## 6. Selection vs tuning

Choose the **family** first, with near-default settings. Tune **after**
(`hyperparameter-tuning` skill). Searching 5 families × 200 trials means you are
optimising the validation set, not the problem.

---

## 7. Anti-patterns

| Anti-pattern | Consequence |
| :--- | :--- |
| Comparing runs on different splits | differences are split noise |
| Reporting the best of 10 seeds | optimistic bias; report the mean |
| Tuning until the test set improves | test set becomes a validation set; your score is a lie |
| Skipping the dummy baseline | you cannot tell learning from a constant |
| Choosing a model on train score | always picks the most overfit |
| Judging by a single metric | see `metrics-and-evaluation` for what the metric hides |
