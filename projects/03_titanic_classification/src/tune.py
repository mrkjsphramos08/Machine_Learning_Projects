"""
Hyperparameter tuning for Project 03 (Titanic Classification).

Uses Stratified 5-Fold Cross-Validation on X_train only (Golden Rule: never tune
against the test set) to find optimal parameters for HistGradientBoostingClassifier.

Usage:
    python -m src.tune
"""

from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.data import load_processed_data


def tune_hist_gradient_boosting(n_iter: int = 35) -> dict:
    """Run a randomized hyperparameter search across 5 stratified folds."""
    print("[*] Loading training data...")
    X_train, _, y_train, _ = load_processed_data()

    print(
        f"[*] Starting RandomizedSearchCV ({n_iter} iterations across 5 stratified folds)..."
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_distributions = {
        "learning_rate": loguniform(0.01, 0.15),
        "max_iter": randint(60, 250),
        "max_depth": [3, 4, 5],
        "min_samples_leaf": randint(10, 30),
        "l2_regularization": [0.0, 0.5, 1.0, 2.5, 5.0],
    }

    search = RandomizedSearchCV(
        estimator=HistGradientBoostingClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_score = search.best_score_
    best_params = search.best_params_

    print("\n" + "=" * 60)
    print(f"[+] Tuning Finished! Best 5-Fold CV ROC-AUC: {best_score:.4f}")
    print("=" * 60)
    print("[+] Optimal Parameters Found:")
    for k, v in sorted(best_params.items()):
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    print("\n[+] To use these optimal parameters, update configs/config.yaml with:")
    print("------------------------------------------------------------")
    print('run_name: "champion_tuned_hist_grad"')
    print('registered_model_name: "TitanicClassifier"')
    print("model:")
    print('  type: "HistGradientBoostingClassifier"')
    print(f"  learning_rate: {best_params['learning_rate']:.4f}")
    print(f"  max_iter: {best_params['max_iter']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  min_samples_leaf: {best_params['min_samples_leaf']}")
    print(f"  l2_regularization: {best_params['l2_regularization']}")
    print("  random_state: 42")
    print("------------------------------------------------------------")

    return best_params


if __name__ == "__main__":
    tune_hist_gradient_boosting()
