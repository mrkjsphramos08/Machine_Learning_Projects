---
name: batch-scoring
description: Scoring files or tables offline with a registry model - loading by alias, enforcing the input contract, writing a stable output schema, wiring a DVC scoring stage, and monitoring predictions produced in batch.
---

# Batch Scoring Skill

Use this when you need predictions for many rows at once: a nightly job, a
backfill, an analyst request. Batch is the cheapest path to real value and needs
no API, no latency budget and no uptime commitment — do it before `fastapi-model-serving`.

---

## 1. Load the model by alias, never by path or run id

```python
import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/diabetes_regression@champion")
```

| URI | Use |
| :--- | :--- |
| `models:/<name>@champion` | **default** — follows promotions automatically |
| `models:/<name>/<version>` | pinning for an audit or a one-off reproduction |
| `runs:/<run_id>/model` | debugging a specific run only |

Hardcoding `models/<file>.pkl` bypasses the registry and silently serves a stale
artifact. `src/paths.py` can still locate a *local* `models/model.pkl` for the
project's own pipeline stages — the registry is for scoring.

---

## 2. Enforce the input contract before predicting

The most common batch failure is a schema change upstream that produces silent
nonsense rather than an error.

```python
import pandas as pd


def load_scoring_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"scoring input is missing columns: {sorted(missing)}")
    return df[required]  # exact column order the model was trained on
```

Column **order** matters for some estimators and for positional signature checks;
selecting an explicit list removes the ambiguity. See `data-validation` for the
full schema treatment.

---

## 3. Write a stable output contract

```python
import datetime as dt


def score(input_path: str, output_path: str, model_uri: str) -> None:
    df = load_scoring_input(input_path)
    model = mlflow.sklearn.load_model(model_uri)

    result = df.copy()
    result["prediction"] = model.predict(df)
    result["model_uri"] = model_uri
    result["scored_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result.to_parquet(output_path, index=False)
```

Always echo: the input key/id column, the prediction, which model produced it, and
when. Without those, downstream consumers cannot trace a bad prediction.

---

## 4. As a DVC stage (cached and incremental)

```yaml
  score:
    cmd: python -m src.score
    deps:
      - src/score.py
      - src/paths.py
      - configs/config.yaml
      - data/scoring/input.csv
    outs:
      - data/predictions/predictions.parquet
```

**Caveat specific to registry models:** DVC hashes files, not MLflow aliases, so a
stage that pulls `@champion` is *not* reproducible from Git alone. Record the
resolved version next to the output:

```python
from mlflow.tracking import MlflowClient

version = (
    MlflowClient().get_model_version_by_alias("diabetes_regression", "champion").version
)
(result_dir / "model_version.json").write_text(f'{{"version": {version}}}')
```

Then the output artefact documents which model produced it, and DVC's cache stays
consistent.

---

## 5. Monitor the batch output (this is your cheapest monitoring)

Predictions produced offline are the natural place to compute drift, before any
serving infrastructure exists:

```python
import pandas as pd

train_sample = pd.read_parquet("data/processed/train.parquet")
scored = pd.read_parquet("data/predictions/predictions.parquet")

print("rows:", len(scored))
print("prediction summary:")
print(scored["prediction"].describe())
print("train target summary:")
print(train_sample["target"].describe())
print("null predictions:", scored["prediction"].isna().sum())
```

Comparisons worth logging every run: row count vs previous run, prediction
distribution vs training target, share of nulls, share of out-of-range values.
See `monitoring-and-drift` for PSI and KS tests.

---

## 6. Practical limits

| Issue | Approach |
| :--- | :--- |
| File too big for memory | `pd.read_csv(..., chunksize=100_000)` and append to Parquet |
| Very large tables | DuckDB or Spark; keep the model load out of the per-chunk loop |
| Slow per-row Python logic | vectorise with pandas before profiling the model |
| Some rows invalid | quarantine them to a rejects file rather than failing the whole job |
| Repeat runs | make the job idempotent: same input → same output path, overwritten |

---

## 7. Checklist

- [ ] Model loaded via registry alias, not a local file path.
- [ ] Input columns asserted, and order fixed explicitly.
- [ ] Output includes id, prediction, model version, scored_at.
- [ ] Output written as Parquet, not CSV (typed, compressed, faster).
- [ ] Reject/quarantine path exists for rows failing validation.
- [ ] Summary statistics printed or logged for later drift comparison.
- [ ] If a DVC stage: the resolved model version is recorded alongside the output.
