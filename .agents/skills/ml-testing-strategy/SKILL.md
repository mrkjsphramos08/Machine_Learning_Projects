---
name: ml-testing-strategy
description: What to test in an ML codebase - the test pyramid for ML, hermetic fixtures with tmp_path and temporary MLflow stores, model invariant tests, and which assertions to avoid because they flake.
---

# ML Testing Strategy Skill

Use this when writing or refactoring tests for a project. In ML, tests do not
verify accuracy; they verify that the *machinery* is sound and that the model
behaves sanely.

---

## 1. The ML test pyramid

| Layer | Example | Speed |
| :--- | :--- | :--- |
| **Unit** | `clean_data()` drops null targets; `build_model()` rejects unknown types | ms |
| **Data contract** | columns, dtypes, ranges, allowed categories (`data-validation`) | ms |
| **Pipeline smoke** | `prepare` then `train` runs end to end and writes its outputs | seconds |
| **Model invariants** | deterministic with a fixed seed, output shape/range, tolerates unseen categories | seconds |
| **Integration** | FastAPI `/predict` returns 200 with a valid payload (`httpx2` needed) | seconds |
| **End-to-end** | `dvc repro` then load the artifact from the registry | slow, run rarely |

Do **not** assert a specific accuracy/RMSE value — that is the definition of a
flaky test.

---

## 2. Hermetic tests (the pattern already in this repo)

Tests must not depend on downloaded data, the real `data/processed/`, or the
shared `mlflow.db`. `projects/_template/tests/test_pipeline.py` shows the pattern:
a `synthetic_project` fixture that writes a small CSV plus a temporary config into
`tmp_path`, pointing `mlflow.tracking_uri` and `mlflow.artifact_root` at temp
locations. Copy that approach:

```python
@pytest.fixture()
def synthetic_project(tmp_path: Path) -> Path:
    ...
    config = {
        "experiment_name": "pytest_<project>_smoke",     # unique per test module
        "mlflow": {
            "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow_test.db').as_posix()}",
            "artifact_root": (tmp_path / "mlruns").as_posix(),
        },
        "paths": {"raw_data": (tmp_path / "data/raw/data.csv").as_posix(), ...},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
```

Verify hermeticity by counting files in the repo's `mlruns/` before and after the
suite — it must not change.

---

## 3. Model invariant tests worth having

```python
import joblib
import pandas as pd
import pytest


def test_predictions_are_deterministic(synthetic_project):
    """Same seed in, same numbers out - the whole point of random_state."""
    prepare_and_save_data(config_path=synthetic_project)
    train_model(config_path=synthetic_project)
    cfg = load_config(synthetic_project)
    X_train, X_test, y_train, y_test = load_processed_data(
        config_path=synthetic_project
    )

    first = joblib.load(cfg["paths"]["model"]).predict(X_test)
    second = joblib.load(cfg["paths"]["model"]).predict(X_test)
    assert (first == second).all()
```

Also worth asserting, adapted to the problem:

| Invariant | Assertion |
| :--- | :--- |
| Output shape/type | `len(preds) == len(X_test)`, numeric dtype |
| Output range | predictions inside the plausible target range |
| Column order invariance | reordering input columns does not change predictions |
| Unseen categories | a category absent from training does not raise |
| Missing values | NaNs in the same places as training do not raise |
| Probability sanity | `predict_proba` rows sum to 1 and stay in [0, 1] |
| Monotonicity | where domain logic demands it (e.g. more spend → not lower risk) |

---

## 4. Assertions to avoid

| Avoid | Use instead |
| :--- | :--- |
| `assert rmse == 53.6546` | `assert rmse < baseline_rmse` or `assert rmse == pytest.approx(53.65, abs=1)` |
| `assert preds == [1, 0, 1]` | property assertions (shape, range, dtype) |
| exact float equality | `pytest.approx` |
| latency thresholds (`< 50ms`) | performance tests outside the unit suite |
| reading real `data/raw/` | the `synthetic_project` fixture |
| sharing one experiment name across tests | a unique `experiment_name` per module |
| asserting library version behaviour | pin versions in `requirements.txt` instead |

---

## 5. Layout and commands

```
projects/<name>/
├── conftest.py                    # puts src/ on sys.path
└── tests/
    └── test_<name>_pipeline.py    # unique basename is optional, not required
```

```powershell
pytest                         # from the project folder: just that project
pytest                         # from the repo root: every project
pytest -q -x                   # stop at the first failure
pytest -k "validation"         # filter by name
pytest --lf                    # rerun only the last failures
pytest -m slow                 # marker-based (declare markers in pytest.ini)
```

`pytest.ini` sets `--import-mode=importlib -ra`, which is what allows every project
to keep its own `tests/test_pipeline.py` without module-name collisions.

---

## 6. Speed budget

Keep the default suite under ~30 seconds. If a test needs a real dataset, a long
training run or the registry, mark it:

```python
@pytest.mark.slow
def test_full_pipeline_on_real_data(): ...
```

and register the marker in `pytest.ini`:

```ini
markers =
    slow: tests that take more than a few seconds
```

---

## 7. Checklist

- [ ] Every pipeline component has at least one smoke or unit test.
- [ ] Data-contract tests assert columns, dtypes and target validity.
- [ ] No test reads `data/raw/` or writes to the repo's `mlruns/`.
- [ ] Model invariants (determinism, shape, range) covered where meaningful.
- [ ] No assertions on exact metric values or timings.
- [ ] Suite under ~30s; slow tests marked and excluded by default in CI.
