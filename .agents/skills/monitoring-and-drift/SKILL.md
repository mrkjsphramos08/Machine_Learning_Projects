---
name: monitoring-and-drift
description: Post-deployment monitoring for ML - service health vs data drift vs performance, PSI and KS drift tests with no extra dependencies, saving a training baseline, delayed-label evaluation, retraining triggers and alert ownership.
---

# Monitoring & Drift Skill

Use this once a model is scoring in production or on a schedule. A deployed model
degrades quietly: nothing crashes, the numbers just stop being true.

---

## 1. Monitor three different things

| Layer | Signal | Detected by |
| :--- | :--- | :--- |
| **Service health** | error rate, latency, timeouts, null predictions | logs/metrics around the scorer |
| **Data drift** | inputs stop resembling training data | PSI / KS tests on feature distributions |
| **Model performance** | the metric on labelled outcomes | delayed-label evaluation |

Performance is the one you actually care about, but labels arrive late (next day,
next month), so drift is the early-warning proxy.

---

## 2. Save a baseline at training time

Drift needs a reference. Persist it when you train, not later:

```python
import pandas as pd

baseline = X_train.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
baseline.to_parquet("data/processed/train_baseline_stats.parquet")
# keep a raw sample too - some tests need the actual distribution, not quantiles
X_train.sample(min(5000, len(X_train)), random_state=42).to_parquet(
    "data/processed/train_baseline_sample.parquet"
)
```

---

## 3. PSI (no extra dependencies)

```python
import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI between a reference sample and a current sample.

    < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 major shift
    """
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # constant reference: PSI is undefined
    ref_share = pd.cut(reference, bins=edges, include_lowest=True).value_counts(normalize=True)
    cur_share = pd.cut(current, bins=edges, include_lowest=True).value_counts(normalize=True)
    ref_share, cur_share = ref_share.align(cur_share, fill_value=0)
    eps = 1e-6  # avoids log(0) and divide-by-zero on empty bins
    ref_share, cur_share = ref_share + eps, cur_share + eps
    return float(((cur_share - ref_share) * np.log(cur_share / ref_share)).sum())
```

Apply per feature and log the results; a single aggregate number hides which
feature moved:

```python
for column in feature_columns:
    psi = population_stability_index(baseline_sample[column], current[column])
    print(f"{column:>12} PSI {psi:.4f}" + ("  <-- investigate" if psi > 0.25 else ""))
```

**Calibration of this implementation** (measured, so you can trust the thresholds):
identical samples give 0.0000; a +0.1 mean shift on a standard normal gives 0.010;
+0.4 gives 0.153; +1.5 gives 1.97. **Do not run it on tiny samples** — with 10
rows it reports 1.22 on data that has not actually drifted. Treat PSI as
meaningless below a few hundred rows per window.

---

## 4. KS test for numeric, chi-square for categorical

```python
from scipy.stats import ks_2samp, chi2_contingency

stat, p_value = ks_2samp(baseline_sample["bmi"], current["bmi"])
print(f"KS={stat:.3f} p={p_value:.4g}")
```

Caveats worth stating out loud: with large samples, p-values become significant
for trivial shifts, so judge by **effect size** (KS statistic), not p alone.
Remember that a drift alert is not a bug report — investigate, do not auto-retrain.

---

## 5. Delayed-label performance evaluation

Log every prediction with its timestamp and key, then join labels when they arrive:

```python
predictions = pd.read_parquet("data/predictions/predictions.parquet")
labels = pd.read_parquet("data/labels/labels.parquet")      # arrives later
joined = predictions.merge(labels[["id", "actual"]], on="id", how="inner")

metric = root_mean_squared_error(joined["actual"], joined["prediction"])
print(f"live RMSE on {len(joined)} labelled rows: {metric:.3f}")
```

Compare that against the frozen holdout score from training. A live score worse
than the holdout score by more than the fold-to-fold std means the world moved.

Beware evaluation skew: the rows that get labelled are usually not a random sample
of the rows you scored.

---

## 6. Optional: Evidently

```powershell
pip install evidently
```

It produces a ready-made drift and data-quality report from two DataFrames, but
its API has changed substantially across releases (0.4 used `Report`/`DataDriftTab`,
0.7 uses `Report` + `presets`). Pin the version you install and check that
version's docs. Everything in §3–§4 above needs no dependency at all, which is why
it is the recommended starting point.

---

## 7. Retraining triggers (decide before you need them)

| Trigger | Example |
| :--- | :--- |
| Drift sustained | PSI > 0.25 on ≥ 3 features for 2 consecutive windows |
| Performance decay | live metric worse than holdout + 1 std |
| Volume/coverage change | share of unseen categories or out-of-range values rising |
| Schedule | quarterly retrain as insurance, regardless of signals |

State the owner and the action for each trigger. An alert nobody owns is noise.

---

## 8. Checklist

- [ ] Training baseline (stats + sample) persisted at train time.
- [ ] Prediction logs include id, features used, prediction and timestamp.
- [ ] PSI (or KS) computed per feature on a schedule.
- [ ] Delayed-label evaluation wired to the prediction log.
- [ ] Thresholds and actions documented, with an owner.
- [ ] Rollback path known (`model-promotion-and-aliases` lists how to move an alias back).
