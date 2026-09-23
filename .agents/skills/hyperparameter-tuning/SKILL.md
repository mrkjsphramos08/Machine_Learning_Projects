---
name: hyperparameter-tuning
description: Procedures for searching hyperparameters with cross-validation - RandomizedSearchCV search-space design, optional Optuna, logging the search to MLflow, budget discipline, and avoiding validation-set overfitting.
---

# Hyperparameter Tuning Skill

Use this **after** the model family is chosen and baselines are established
(`baseline-and-model-selection`), never before. Tuning a high-variance model just
finds the most overfit setting faster.

---

## 1. Tune on cross-validation, not on the holdout

The holdout is your report card; tuning against it turns it into a validation set
and your final number becomes optimistic. Search on `X_train`, then confirm once.

---

## 2. Randomised search beats grid search

For more than about three parameters, a grid explodes and wastes runs on
irrelevant dimensions. Sample instead:

```python
from scipy.stats import loguniform, randint
from sklearn.model_selection import KFold, RandomizedSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=42)     # match your protocol

param_dist = {
    "est__n_estimators": randint(100, 800),
    "est__max_depth": randint(3, 20),
    "est__min_samples_leaf": randint(1, 20),
    "est__max_features": [None, "sqrt", "log2"],
}

search = RandomizedSearchCV(
    pipeline,                       # the full Pipeline, including preprocessing
    param_dist,
    n_iter=40,                      # ~40-60 is plenty for a first pass
    cv=cv,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)
print(-search.best_score_, search.best_params_)
best_pipeline = search.best_estimator_        # already refit on all of X_train
```

The `est__` prefix addresses the pipeline step named `est` — the same naming used
in the `feature-engineering` skill. Tuning without the step prefix silently fails.

---

## 3. Search-space design

| Parameter | Distribution | Reasoning |
| :--- | :--- | :--- |
| `n_estimators` | `randint(100, 800)` | plateaus; more is rarely harmful, just slower |
| `max_depth` | `randint(3, 20)` | the main capacity knob |
| `min_samples_leaf` | `randint(1, 20)` | regularisation for trees |
| `max_features` | `[None, "sqrt", "log2"]` | decorrelation vs speed |
| `learning_rate` (boosting) | `loguniform(1e-3, 0.3)` | **log-uniform**, not linear |
| `n_estimators` (boosting) | `randint(100, 1000)` | pairs with learning rate |
| `C` (linear/logistic) | `loguniform(1e-3, 1e3)` | spans orders of magnitude |

Rule: use `loguniform` for anything multiplicative (rates, regularisation
strengths), `randint` for counts, and a small explicit list for categorical options.

---

## 4. Optional: Optuna (TPE + pruning)

```powershell
pip install optuna
```

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }
    scores = cross_validate(make_pipeline(params), X_train, y_train, cv=cv,
                            scoring="neg_root_mean_squared_error")
    return scores["test_score"].mean()

study = optuna.create_study(direction="maximize")   # maximise neg-RMSE
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(study.best_value, study.best_params)
```

Optuna wins when trials are few and expensive, or when you want pruning of bad
trials. `RandomizedSearchCV` wins on simplicity and zero new dependencies.

---

## 5. Log the search to MLflow (repo convention)

The project's `train_model()` logs one run per configuration. After a search,
either:

- **Preferred:** write the winning values into `configs/config.yaml`, bump
  `run_name` (e.g. `baseline_rf` → `tuned_rf_v1`), and re-run `python -m src.train`
  so the run is reproducible from the config; or
- Log the search itself as one run with `mlflow.log_params(study.best_params)`
  plus `mlflow.log_metric("cv_best_score", best_value)` and extra params like
  `n_iter` and `cv_folds` so you can tell tuned runs from hand-picked ones.

Never overwrite a config value without a run that records the change.

---

## 6. Budget and the winner's curse

- Scores within one standard deviation of each other are **ties**.
- With 200 trials, the best CV score is optimistically biased — the search found
  a lucky fold. Confirm the winner once on the frozen holdout.
- Stop when improvements are inside the noise band; you are then just fitting the
  validation noise.
- Keep the search reproducible: fixed `random_state` and a recorded search space.

---

## 7. Checklist

- [ ] Baselines and the model family are already decided.
- [ ] CV splitter and seed match the project's protocol.
- [ ] The tuned object is the full `Pipeline` (preprocessing included), so
      `best_estimator_` is directly usable and registrable.
- [ ] `n_iter`/`n_trials` kept modest for a first pass.
- [ ] Winner written to `configs/config.yaml` with a new `run_name`.
- [ ] Final number taken from the holdout, once.
