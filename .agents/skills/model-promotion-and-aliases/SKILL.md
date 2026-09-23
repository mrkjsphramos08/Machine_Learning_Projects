---
name: model-promotion-and-aliases
description: Champion/challenger comparison and release gating with MLflow model aliases - promotion criteria, alias management replacing deprecated registry stages, model signatures, and rollback.
---

# Model Promotion & Aliases Skill

Use this when a candidate model beats the incumbent and you need a defensible,
reversible way to make it the one serving traffic.

---

## 1. Aliases, not stages (verified against MLflow 3.16.0)

Registry **stages** (`Staging`/`Production`) still run but are deprecated:

> `transition_model_version_stage` is deprecated since 2.9.0. Model registry
> stages will be removed in a future major release.

Use **aliases** instead — verified working, including loading by alias URI:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name="diabetes_regression", alias="champion", version=3
)

import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/diabetes_regression@champion")  # works
```

Suggested alias vocabulary:

| Alias | Meaning |
| :--- | :--- |
| `champion` | the version serving production |
| `challenger` | the candidate under evaluation |
| `latest` | most recently registered (diagnostics only — never serve from it) |

Aliases are movable pointers: promotion and rollback are one API call and need no
code deploy.

---

## 2. Write the gate down before you look at the numbers

A promotion policy that lives only in your head will bend. State it in the project
README:

| Gate | Example |
| :--- | :--- |
| Primary metric | RMSE strictly better than champion on the **same frozen holdout** |
| Margin | improvement must exceed fold-to-fold std (otherwise it is a tie) |
| Worst slice | no segment may degrade by more than X% |
| Data contract | validation tests pass (`data-validation`) |
| Artefacts | model logged with a signature and input example |
| Reversibility | previous version stays registered and tagged |

---

## 3. Champion/challenger procedure

```powershell
cd projects\01_diabetes_regression

# 1. Train the candidate: change run_name (and hyperparameters) in configs/config.yaml
python -m src.train            # e.g. run_name: challenger_rf_v2

# 2. Register it: set model.registered_model_name in configs/config.yaml
#    (train.py passes registered_model_name through to mlflow.sklearn.log_model)

# 3. Mark it as the challenger
```

```python
client.set_registered_model_alias(
    name="diabetes_regression", alias="challenger", version=3
)
```

Then evaluate **both versions on the identical holdout** — never compare a new
number against an old one measured on different rows:

```python
import mlflow.sklearn
from src.data import load_config, load_processed_data
from src.train import evaluate_model

X_train, X_test, y_train, y_test = load_processed_data("configs/config.yaml")
for alias in ("champion", "challenger"):
    model = mlflow.sklearn.load_model(f"models:/diabetes_regression@{alias}")
    print(alias, evaluate_model(model, X_test, y_test))
```

Promote only when every gate in §2 passes:

```python
version = client.get_model_version_by_alias("diabetes_regression", "challenger").version
client.set_model_version_tag(
    "diabetes_regression",
    version,
    "promoted_reason",
    "rmse 51.2 -> 48.9 on frozen holdout",
)
client.set_registered_model_alias("diabetes_regression", "champion", version)
client.delete_registered_model_alias("diabetes_regression", "challenger")
```

---

## 4. Log a signature so serving can validate input

```python
import mlflow
from mlflow.models import infer_signature

signature = infer_signature(X_train, model.predict(X_train.head(5)))
mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    signature=signature,
    input_example=X_train.head(3),
    registered_model_name="diabetes_regression",
)
```

The signature turns "the API accepted a nonsense payload and returned a number"
into a validation error, and it documents the exact expected columns.

---

## 5. Rollback

```python
previous = 2  # the version you replaced
client.set_registered_model_alias("diabetes_regression", "champion", previous)
```

Because rollback is an alias move, keep the previous champion registered and
tagged forever. Aliases are cheap; retraining is not.

---

## 6. Anti-patterns

| Anti-pattern | Why it hurts |
| :--- | :--- |
| Promoting on the test set you keep re-using | the score is no longer an estimate |
| Promoting on a single aggregate metric | hidden slice regressions ship silently |
| Serving `@latest` | untested versions reach users |
| No rollback target | incidents become retraining projects |
| Comparing champion and challenger on different data | the comparison is meaningless |
| Overwriting the model artifact instead of registering a new version | history is destroyed |

---

## 7. Checklist

- [ ] Promotion policy written in the project README before evaluating.
- [ ] Both models scored on the identical frozen holdout.
- [ ] Slices checked, not just the aggregate.
- [ ] Model logged with signature + input example.
- [ ] Version tagged with the promotion reason.
- [ ] Rollback version identified and still registered.
