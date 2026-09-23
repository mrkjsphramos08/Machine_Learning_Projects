---
name: model-interpretability
description: Techniques for explaining model behaviour and diagnosing failures - permutation importance, partial dependence, optional SHAP, residual analysis, and per-segment error slicing.
---

# Model Interpretability & Error Analysis Skill

Use this to answer two questions: *why* does the model predict this, and *where*
is it failing. Error analysis usually improves a model more than tuning does.

---

## 1. Order of operations (cheapest first)

1. **Error analysis / slices** — no libraries, biggest payoff.
2. **Permutation importance** — model-agnostic, `scikit-learn` only.
3. **Partial dependence / ICE** — shape of each effect.
4. **SHAP** — only when you need per-prediction attribution and can afford it.

---

## 2. Permutation importance (start here)

```python
import pandas as pd
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_root_mean_squared_error",
)
importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values(
    ascending=False
)
print(importance.head(15))
```

How to read it: importance is the drop in score when a column is shuffled.
Caveats that matter:

- Correlated features **share** importance — if two columns carry the same signal,
  both look unimportant when shuffled independently.
- If you score on training data, a memorising feature looks harmless; always use
  the holdout.
- A high-importance ID-like column is a **leakage alarm**, not a finding.

Tree `feature_importances_` (impurity-based) is faster but biased toward
high-cardinality features — use it for a quick look only.

---

## 3. Partial dependence

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_test, ["bmi", "s5"], grid_resolution=50
)
```

Shows the marginal effect of a feature. Unreliable when features are strongly
correlated (it averages over impossible feature combinations).

---

## 4. Residual analysis (regression)

```python
import pandas as pd

err = pd.DataFrame({"y_true": y_test, "y_pred": model.predict(X_test)})
err["residual"] = err["y_true"] - err["y_pred"]

import matplotlib.pyplot as plt

plt.scatter(err["y_pred"], err["residual"], s=8, alpha=0.6)
plt.axhline(0, color="red", linewidth=1)
plt.xlabel("prediction")
plt.ylabel("residual")
plt.show()
```

| Pattern | Meaning | Next move |
| :--- | :--- | :--- |
| Random scatter around 0 | model captures the signal | stop, ship it |
| Curved band | missed non-linearity | add features / use a different family |
| Funnel shape | error grows with magnitude | model the log target, or MAE |
| Clusters / bands | hidden groups | inspect the segments, maybe add the group as a feature |
| Heavy tail on one side | systematic over/under-prediction | check for a censored target |

---

## 5. Sliced error analysis (works for both tasks)

The highest-value diagnostic: compute the metric per segment.

```python
import numpy as np
from sklearn.metrics import root_mean_squared_error

slices = err.assign(**{"bucket": err["y_pred"].round(-1)})
slices.groupby("bucket").agg(
    n=("residual", "size"),
    mean_resid=("residual", "mean"),
    rmse=("residual", lambda r: float(np.sqrt((r**2).mean()))),
)
```

Do the same for categorical features (`groupby("sex")`, `groupby("region")`). A
segment with 5× the error of the rest is either a data problem, a coverage
problem, or a fairness problem — all worth knowing before release.

For classification, read the confusion matrix cell by cell and inspect the
misclassified rows; look for a pattern in the raw text/columns.

---

## 6. Optional: SHAP

```powershell
pip install shap
```

```python
import shap

explainer = shap.TreeExplainer(model.named_steps["est"])  # tree models
sample = X_test.sample(min(500, len(X_test)), random_state=42)
explanation = explainer(sample)  # shap >= 0.45
explanation.summary_plot()  # not available in all versions
```

Notes so you are not surprised:
- SHAP's API has changed across releases (`shap_values(...)` vs `explainer(...)`
  returning `Explanation` objects). Check the installed version's docs.
- Sample rows for speed; exact SHAP on 100k rows is slow.
- If the model is inside a `Pipeline`, explain the estimator and explain against
  the **transformed** feature matrix (`model.named_steps["prep"].transform(X_test)`),
  otherwise the plot labels will not match.

---

## 7. Where interpretation output belongs

| Output | Home |
| :--- | :--- |
| Exploratory plots and reasoning | `notebooks/02_interpret.ipynb` |
| A stable table of importances | `reports/` (gitignored) or an MLflow artifact |
| Decisions ("drop `passengerid`", "log the target") | the config and the project README |

Do not put interpretation inside the training pipeline — it is analysis, not
training.

---

## 8. Red flags this skill is designed to catch

| Red flag | Likely cause |
| :--- | :--- |
| One feature dominates importance | target leakage |
| Very high score at first attempt | leakage, or a duplicate of the target |
| Residuals curve sharply | missing non-linearity / wrong family |
| One slice far worse | coverage or label-noise problem in that segment |
| Feature importance contradicts domain knowledge | ask a domain expert before shipping |
