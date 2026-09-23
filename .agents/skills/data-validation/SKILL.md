---
name: data-validation
description: Turning dataset expectations into executable checks - zero-dependency pytest validation and optional pandera schemas, validating at ingestion and before training, and cheap train/test consistency checks.
---

# Data Validation Skill

Use this when a dataset is new, when a pipeline silently produces garbage, or
when you want a failing build instead of a silently wrong model. `.agents/rules/
mlops_standards.md §4` already requires validation tests; this skill says what to
assert and where to put it.

---

## 1. Validate in two places

| Where | What it protects |
| :--- | :--- |
| **At ingestion** — start of the `prepare` stage | the raw file is what you think it is (schema changed upstream, wrong file downloaded) |
| **Before training** — in `tests/` | your own preprocessing output keeps the contract (a column vanished, dtypes changed) |

Fail loudly at ingestion. A schema change discovered after training costs a day.

---

## 2. Zero-dependency version (works today, no install)

Put this in `src/data.py` and call it from `prepare_and_save_data()`:

```python
import pandas as pd


def validate_raw_data(df: pd.DataFrame, target_column: str) -> None:
    """Fail fast when the raw file does not match the documented contract."""
    required = {"age", "sex", "bmi", "bp", target_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"raw data is missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("raw data has zero rows")

    if df[target_column].isna().any():
        raise ValueError(f"target '{target_column}' contains null values")

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise TypeError(
            f"target '{target_column}' must be numeric, got {df[target_column].dtype}"
        )

    if df[target_column].nunique() < 2:
        raise ValueError("target has a single distinct value - nothing to learn")
```

And the same expectations as *tests* in `tests/test_<name>_pipeline.py`:

```python
def test_raw_data_contract(synthetic_project):
    prepare_and_save_data(config_path=synthetic_project)
    X_train, X_test, y_train, y_test = load_processed_data(
        config_path=synthetic_project
    )

    assert list(X_train.columns) == FEATURES  # exact column list and order
    assert len(X_train) > 0 and len(X_test) > 0
    assert not y_train.isna().any()
    assert y_train.nunique() > 1
    assert pd.api.types.is_numeric_dtype(y_train)
```

This mirrors the tests already shipped in `projects/_template`.

---

## 3. Optional: declarative schemas with pandera

```powershell
pip install pandera
```

```python
import pandera as pa
from pandera import Check, Column, DataFrameSchema

raw_schema = DataFrameSchema(
    {
        "age": Column(float, Check.between(0, 120), nullable=False),
        "sex": Column(str, Check.isin(["male", "female", "unknown"])),
        "bmi": Column(float, Check.gt(0), nullable=True),
        "target": Column(float, Check.between(0, 1000), nullable=False),
    },
    strict=False,  # True = extra columns are an error
    coerce=True,  # attempt dtype coercion
)

raw_schema.validate(
    df, lazy=True
)  # lazy=True collects ALL violations, not just the first
```

Notes:
- API shown is pandera 0.19+ (`DataFrameSchema`). Newer releases also offer a
  class-based `DataFrameModel`; `pip show pandera` and check the version docs if
  something does not resolve.
- `lazy=True` matters: you want the full list of problems in one run.
- Keep `strict=False` while exploring, switch it on once the contract is stable.

Wrap it so the pipeline calls one function:

```python
def validate_raw_data(df: pd.DataFrame) -> None:
    raw_schema.validate(df, lazy=True)
```

---

## 4. Train/test consistency (cheap, high value)

The most common silent failure is a test file that differs from the train file.
Compare them explicitly, in the `prepare` stage or a test:

```python
def check_split_consistency(train_df, test_df, target_column: str) -> None:
    assert list(train_df.columns) == list(test_df.columns), "train/test column mismatch"
    features = [c for c in train_df.columns if c != target_column]
    for col in features:
        train_set = set(train_df[col].dropna().unique()[:1000])
        test_set = set(test_df[col].dropna().unique()[:1000])
        unseen = test_set - train_set
        if unseen:
            print(f"[!] {col}: {len(unseen)} value(s) in test never seen in train")
```

Unseen categories are not automatically an error — but with
`OneHotEncoder(handle_unknown="ignore")` they become all-zero rows, so you want to
know about them.

---

## 5. What NOT to validate

| Bad assertion | Why |
| :--- | :--- |
| `len(df) == 10_000` | upstream row counts legitimately change |
| mean/std within tight bounds | distributions drift; alerts become noise |
| exact float equality | floating point |
| "no missing values" when the pipeline imputes | validates the wrong thing |
| a specific model score | that is a model test, and it will flake |

Validate **structure and meaning** (columns, dtypes, ranges, allowed categories),
not incidental statistics.

---

## 6. Checklist

- [ ] `validate_raw_data()` runs at the start of the `prepare` stage.
- [ ] The contract lives with the code, and the column list is documented in the README.
- [ ] Validation tests exist in `tests/` and pass on synthetic fixtures.
- [ ] Failure messages name the offending column and reason.
- [ ] `dvc repro` fails loudly rather than producing a bad `train.parquet`.
