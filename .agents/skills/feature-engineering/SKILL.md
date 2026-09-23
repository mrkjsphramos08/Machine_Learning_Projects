---
name: feature-engineering
description: Building leak-free preprocessing and feature pipelines with sklearn Pipeline and ColumnTransformer - imputation, encoding, scaling, datetime and text features, fit-on-train-only discipline, and persisting the whole pipeline as the model artifact.
---

# Feature Engineering & Preprocessing Skill

Use this whenever raw data has missing values, categorical columns, mixed scales,
or datetime/text fields — i.e. for essentially every real dataset. The one idea
that matters: preprocessing must be **fitted on the training split only**, and it
must travel with the model.

---

## 1. Where preprocessing lives in this repo

| Artifact | Location |
| :--- | :--- |
| Column lists | `projects/<name>/configs/config.yaml` (never literals in Python) |
| Frame-level cleaning (dedupe, dtype fixes, drop leakage columns) | `src/data.py`, inside the `prepare` stage |
| `Pipeline` / `ColumnTransformer` construction | `src/train.py` (or a new `src/features.py` imported by it) |
| What gets persisted and registered | the **whole pipeline**, not the bare estimator |

Do frame-level cleaning in the `prepare` stage so the leak-free transform is the
only thing that stays dynamic between train and test.

---

## 2. Declare your columns in the config

```yaml
features:
  numeric: ["age", "fare", "sibsp", "parch"]
  categorical: ["sex", "embarked", "pclass"]
  drop: ["passengerid", "name", "ticket", "cabin"]
```

This keeps `src/` free of literals and makes the feature set reviewable in a diff.

---

## 3. Build the transformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
```

Why each choice: the imputer sits *inside* the pipeline so medians are learned on
train only; `handle_unknown="ignore"` stops unseen categories crashing inference;
`remainder="drop"` makes dropping unlisted columns explicit rather than accidental.

---

## 4. Train the pipeline as one unit

```python
model = Pipeline(
    [
        (
            "prep",
            build_preprocessor(
                cfg["features"]["numeric"], cfg["features"]["categorical"]
            ),
        ),
        ("est", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)
model.fit(X_train, y_train)
```

`evaluate_model()` in `src/train.py` needs no change — it calls `model.predict()`.
Because the whole pipeline is `joblib.dump`-ed and logged with
`mlflow.sklearn.log_model`, serving gets the identical preprocessing for free.
The classic production bug this prevents: training with scaled features, serving
with raw ones.

---

## 5. Per-column-type menu

| Column type | Recommended | Watch out for |
| :--- | :--- | :--- |
| numeric with NaN | `SimpleImputer(strategy="median")` | median is robust to outliers; mean is not |
| heavily skewed numeric | `log1p` / `QuantileTransformer` | if you transform the **target**, invert predictions before scoring |
| low-cardinality categorical | `OneHotEncoder(handle_unknown="ignore")` | without `handle_unknown`, serving raises on new categories |
| ordinal categorical | `OrdinalEncoder(categories=[...])` | never rely on alphabetical order |
| high-cardinality categorical | `TargetEncoder` (available in sklearn 1.9) or frequency encoding | must live **inside** the pipeline — it uses the target |
| datetime | sin/cos of day-of-year, hour, `is_weekend`, `days_since_epoch` | raw timestamps let the model memorize |
| free text | `TfidfVectorizer` in its own branch | keep it out of the numeric branch |
| ID-like column | drop it | usually pure leakage |

---

## 6. Leakage checklist

- [ ] Split **first**, then fit preprocessing on `X_train` only.
- [ ] No `.fit()` or `fit_transform()` calls outside a `Pipeline`.
- [ ] Target encoders only ever inside the pipeline.
- [ ] Imputation / scaling statistics not computed on the full dataset.
- [ ] Any feature that is unavailable at prediction time removed from `X`.
- [ ] The `drop` list in the config was actually honoured (`remainder="drop"`).

---

## 7. Verify it

```python
model.named_steps  # structure is what you think it is
model.named_steps["prep"].get_feature_names_out()  # expanded column names
```

Round-trip test (catches pickling and version problems):

```python
import joblib

joblib.dump(model, "models/tmp.pkl")
reloaded = joblib.load("models/tmp.pkl")
assert (reloaded.predict(X_test[:5]) == model.predict(X_test[:5])).all()
```

`pytest` in the project already exercises the pipeline end to end; extend
`tests/test_<name>_pipeline.py` with an invariant like "predictions do not change
when column order in the input frame changes".
