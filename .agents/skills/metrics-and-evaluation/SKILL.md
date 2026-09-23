---
name: metrics-and-evaluation
description: How to choose and compute evaluation metrics for regression and classification - the accuracy trap, precision/recall/F1, ROC-AUC vs PR-AUC, thresholds, class imbalance, calibration, and extending evaluate_model() in a project.
---

# Metrics & Evaluation Skill

Use this when choosing the number you will optimise, and whenever a score looks
"too good". Choosing the wrong metric is the most expensive mistake in ML because
nothing downstream will catch it.

---

## 1. Derive the metric from the decision

| Question you are answering | Metric |
| :--- | :--- |
| "How wrong are we, in the target's units?" | RMSE (or MAE if outliers are legitimate) |
| "How often are we wrong?" | MAE |
| "Percentage error matters to stakeholders" | MAPE (only if target > 0) |
| "Are we better than predicting the mean?" | R² (and compare against the dummy baseline) |
| "Is this a rare event?" (fraud, churn, disease) | PR-AUC / recall, **not** accuracy |
| "Do we need to rank, not threshold?" | ROC-AUC |
| "Are the predicted probabilities trustworthy?" | Brier score / calibration curve |
| "What is the cost of each error type?" | a cost-weighted metric via `make_scorer` |

If you cannot name the decision this model supports, you cannot choose a metric.

---

## 2. Regression metrics

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

rmse = root_mean_squared_error(y_test, y_pred)  # sklearn >= 1.4 (you have 1.9)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

| Metric | Reads as | Fails when |
| :--- | :--- | :--- |
| RMSE | same units as target; punishes large errors | a few outliers dominate the number |
| MAE | same units; the typical error | large errors look harmless |
| MAPE | percentage error | target has zeros or near-zeros |
| R² | improvement over predicting the mean | interpreted as a percentage; negative R² is possible and means "worse than the mean" |

Always put the **dummy baseline's** RMSE next to yours. "RMSE 53.6 on a target
ranging 25–346" only means something compared to the mean-predictor.

---

## 3. Classification metrics

```python
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # column 1 = positive class
```

**The accuracy trap.** With 95% negatives, predicting "negative" always scores
95% accuracy and is useless. Check `df[y].value_counts(normalize=True)` first.

| Metric | Prefer it when |
| :--- | :--- |
| Precision | false positives are expensive (spam filter) |
| Recall | false negatives are expensive (cancer screening) |
| F1 | you need one number and both errors matter |
| ROC-AUC | ranking quality, roughly balanced classes |
| PR-AUC (`average_precision_score`) | rare positives — the honest choice when imbalanced |

**Thresholds.** `predict()` uses 0.5, which is arbitrary. If the cost of the two
error types differs, choose the threshold on **validation** data, not test:

```python
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    print(t, "f1:", round(f1_score(y_val, preds), 4))
```

**Imbalance handling, cheapest first**
1. `class_weight="balanced"` on the estimator (one line, no data change)
2. choose a threshold deliberately (§ above)
3. resample (`imbalanced-learn`) — **inside** CV folds only, never before splitting
4. `CalibratedClassifierCV` if you need probabilities, not just decisions

---

## 4. Extending `evaluate_model()` in a project

`src/train.py` ships with regression metrics. For classification, return a dict
with stable keys so MLflow history stays comparable:

```python
def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        metrics["pr_auc"] = average_precision_score(y_test, y_prob)
    return {k: float(v) for k, v in metrics.items()}
```

`mlflow.log_metrics(metrics)` already handles the dict, and the project's
`pytest` asserts the key set — so update the assertion when you change the keys.

---

## 5. Reporting rules

- Report the **baseline** alongside every score.
- Report **mean ± std** across folds, not a single split.
- Compute the final number on the frozen holdout **once**.
- Never compare metrics computed on different row sets.
- If you log-transform the target, invert predictions before scoring.

---

## 6. Scoring bugs that hide real failures

| Bug | Effect |
| :--- | :--- |
| Accuracy on imbalanced data | looks great, predicts one class |
| Tuning the threshold on the test set | optimistic, unreproducible |
| Comparing RMSE across different test subsets | meaningless difference |
| `predict_proba` column 0 instead of 1 | silently inverted metrics |
| Forgetting `zero_division=0` | crashes or warns on empty classes |
| R² reported as a percentage | 0.45 is not 45% correct, it is 45% of variance explained |
