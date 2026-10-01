"""Model diagnostics: Overfitting check, 5-Fold Cross-Validation, and Residual Analysis.

Run with: python -m src.evaluate
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate

from src.data import load_processed_data
from src.paths import PROJECT_ROOT


def run_diagnostics(config_path: str | Path = "configs/config.yaml") -> None:
    print("=" * 65)
    print(" MODEL DIAGNOSTIC & EVALUATION SUITE")
    print("=" * 65)

    X_train, X_test, y_train, y_test = load_processed_data(config_path)
    model_path = PROJECT_ROOT / "models" / "model.pkl"

    if not model_path.exists():
        print(f"[!] Model file not found at {model_path}. Train the model first.")
        return

    model = joblib.load(model_path)

    # ---------------------------------------------------------
    # 1. Overfitting Check: Train vs Test Scores
    # ---------------------------------------------------------
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_rmse = root_mean_squared_error(y_train, train_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    print("\n--- 1. OVERFITTING CHECK (TRAIN vs. TEST) ---")
    print(f"Train RMSE: {train_rmse:.4f}  |  Test RMSE: {test_rmse:.4f}")
    print(f"Train R2:   {train_r2:.4f}  |  Test R2:   {test_r2:.4f}")
    print(f"R2 Generalization Gap: {(train_r2 - test_r2):.4f}")

    if (train_r2 - test_r2) > 0.10:
        print(
            "[!] Warning: High variance detected (Train R2 is much higher than Test R2)."
        )
    else:
        print("[+] Good: Generalization gap is healthy.")

    # ---------------------------------------------------------
    # 2. 5-Fold Cross-Validation Stability
    # ---------------------------------------------------------
    print("\n--- 2. 5-FOLD CROSS-VALIDATION (STABILITY TEST) ---")
    print("[*] Running 5-fold CV on training data...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        return_train_score=True,
        n_jobs=-1,
    )

    cv_val_rmse_mean = -scores["test_rmse"].mean()
    cv_val_rmse_std = scores["test_rmse"].std()
    cv_val_r2_mean = scores["test_r2"].mean()
    cv_val_r2_std = scores["test_r2"].std()

    print(
        f"Validation RMSE across 5 folds: {cv_val_rmse_mean:.4f} +/- {cv_val_rmse_std:.4f}"
    )
    print(
        f"Validation R2 across 5 folds:   {cv_val_r2_mean:.4f} +/- {cv_val_r2_std:.4f}"
    )
    print(
        f"Confidence interval: R2 is consistently between {cv_val_r2_mean - 2 * cv_val_r2_std:.4f} and {cv_val_r2_mean + 2 * cv_val_r2_std:.4f}"
    )

    # ---------------------------------------------------------
    # 3. Residual & Error Analysis
    # ---------------------------------------------------------
    residuals = (
        y_test - test_preds
    )  # Positive = underpredicted, Negative = overpredicted
    print("\n--- 3. RESIDUAL ANALYSIS (ERROR DISTRIBUTION) ---")
    print(
        f"Mean Error (Systematic Bias): {residuals.mean():.4f} (Near 0 means unbiased)"
    )
    print(f"Median Absolute Error:        {np.median(np.abs(residuals)):.4f}")
    print(f"Max Under-prediction:         +{residuals.max():.4f} ($100k units)")
    print(f"Max Over-prediction:          {residuals.min():.4f} ($100k units)")

    # Slice by artificial census cap ($500k = 5.0)
    capped_mask = y_test >= 4.99
    rmse_uncapped = root_mean_squared_error(
        y_test[~capped_mask], test_preds[~capped_mask]
    )
    rmse_capped = root_mean_squared_error(y_test[capped_mask], test_preds[capped_mask])
    print(f"\n  [Slice] RMSE on Normal Homes (< $500k): {rmse_uncapped:.4f}")
    print(f"  [Slice] RMSE on Capped Homes (= $500k): {rmse_capped:.4f}")
    print("  -> Notice how the $500k census ceiling skews error upwards!")

    # ---------------------------------------------------------
    # 4. Permutation Feature Importance
    # ---------------------------------------------------------
    print("\n--- 4. PERMUTATION FEATURE IMPORTANCE (TEST SET) ---")
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    importance_series = pd.Series(
        perm.importances_mean, index=X_test.columns
    ).sort_values(ascending=False)

    for rank, (feature, drop) in enumerate(importance_series.items(), 1):
        print(f"  {rank:2d}. {feature:<18} (score drop: {drop:.4f})")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    run_diagnostics()
