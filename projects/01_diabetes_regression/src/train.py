"""
Model training and evaluation with MLflow experiment tracking.

================================================================================
MLflow calls used here (and why):
================================================================================
1. mlflow.set_tracking_uri(...):
   MLflow 3.x stores runs in SQLite (./mlflow.db) by DEFAULT, resolved relative
   to the CURRENT WORKING DIRECTORY. Running from a project folder would
   silently create a second database there and split your history. We therefore
   pin the URI to the single repo-level mlflow.db (see configure_tracking()).

2. mlflow.set_experiment(name):
   Groups runs under a named banner. Each PROJECT must use a distinct name so
   that a shared mlflow.db still yields clean, comparable experiment groups.

3. mlflow.start_run(run_name=...):
   Opens one tracked execution and captures metadata (start time, user, git
   commit). The `with` form guarantees the run is marked FINISHED or FAILED
   even when an exception is raised mid-training.

4. mlflow.log_params / log_metrics:
   Inputs (hyperparameters, seeds, split sizes) must be immutable and are logged
   once; metrics are outputs and may be logged repeatedly over time.

5. mlflow.sklearn.log_model(...):
   Serializes the model together with an environment spec so it can be loaded
   later by run ID / registry URI for serving or batch inference.
================================================================================

Run as a DVC stage:   python -m src.train
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_config, load_processed_data
from src.paths import PROJECT_ROOT, REPO_ROOT, ensure_parent, resolve

DEFAULT_CONFIG_PATH = "configs/config.yaml"


def configure_tracking(config: dict) -> str:
    """Point MLflow at the shared repo-level SQLite store and return the URI.

    A project may override the store by setting `mlflow.tracking_uri` in its
    config (this is what tests use to stay fully isolated).
    """
    tracking_uri = (config.get("mlflow") or {}).get("tracking_uri")
    if not tracking_uri:
        tracking_uri = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def ensure_experiment(experiment_name: str, config: dict) -> str:
    """Create the MLflow experiment if needed, pinned to a repo-level artifact root.

    MLflow derives a NEW experiment's `artifact_location` from the current
    working directory, so a project run through DVC would otherwise scatter its
    model artifacts into `projects/<name>/mlruns/`. Pinning the location here
    keeps every project's artifacts in the one shared store at the repo root.

    Returns the experiment id.
    """
    mlflow_config = config.get("mlflow") or {}
    artifact_root = mlflow_config.get("artifact_root")
    if artifact_root:
        root_path = Path(artifact_root)
        if not root_path.is_absolute():
            root_path = PROJECT_ROOT / root_path
    else:
        # Default: one artifact folder per experiment, e.g. mlruns/01_diabetes_regression
        root_path = REPO_ROOT / "mlruns" / experiment_name

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=root_path.as_uri()
        )
        print(f"[*] Created experiment '{experiment_name}' -> artifacts in {root_path}")
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def build_model(model_cfg: dict):
    """Instantiate the estimator described by the config's `model` section."""
    model_type = model_cfg.get("type", "RandomForestRegressor")
    if model_type == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 6),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            random_state=model_cfg.get("random_state", 42),
        )
    elif model_type == "Ridge":
        return Ridge(
            alpha=model_cfg.get("alpha", 1.0),
            random_state=model_cfg.get("random_state", 42),
        )
    elif model_type == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(
            max_iter=model_cfg.get("max_iter", 100),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            max_depth=model_cfg.get("max_depth", 6),
            random_state=model_cfg.get("random_state", 42),
        )
    raise ValueError(
        f"Unsupported model type '{model_type}'. "
        "Supported types: 'RandomForestRegressor', 'Ridge', 'HistGradientBoostingRegressor'."
    )


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Compute regression metrics on the held-out test set.

    Swap in the metric that actually matches your problem (e.g. ROC-AUC and F1
    for classification, MAPE for skewed regression targets).
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def train_model(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, float]:
    """Train, evaluate, log to MLflow and persist the model. Returns the metrics."""
    config = load_config(config_path)

    experiment_name = config.get("experiment_name", "default_experiment")
    run_name = config.get("run_name", "run")
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    paths_cfg = config.get("paths", {})

    tracking_uri = configure_tracking(config)
    ensure_experiment(experiment_name, config)

    with mlflow.start_run(run_name=run_name):
        print(
            f"[*] Run '{run_name}' | experiment '{experiment_name}' | store {tracking_uri}"
        )

        X_train, X_test, y_train, y_test = load_processed_data(config_path)

        flat_params = {
            "test_size": dataset_cfg.get("test_size", 0.2),
            "data_random_state": dataset_cfg.get("random_state", 42),
            "model_type": model_cfg.get("type", "RandomForestRegressor"),
            "model_random_state": model_cfg.get("random_state", 42),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "n_features": X_train.shape[1],
        }
        flat_params.update(
            {
                k: v
                for k, v in model_cfg.items()
                if k not in {"type", "random_state"}
                and isinstance(v, (int, float, str))
            }
        )
        mlflow.log_params(flat_params)

        model = build_model(model_cfg)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        model_path = ensure_parent(resolve(paths_cfg.get("model", "models/model.pkl")))
        joblib.dump(model, model_path)

        signature = infer_signature(X_train, model.predict(X_train.iloc[:5]))
        input_example = X_train.iloc[:5]
        registered_name = model_cfg.get("registered_name")

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_name,
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        print("[+] Training completed successfully!")
        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    MAE:  {metrics['mae']:.4f}")
        print(f"    R2:   {metrics['r2']:.4f}")
        print(f"    Model written to: {model_path}")
        print("[+] Metrics + artifact logged to MLflow.")

        # Champion / Challenger Gate
        if registered_name and model_info.registered_model_version is not None:
            client = MlflowClient()
            new_version = model_info.registered_model_version

            try:
                champion_mv = client.get_model_version_by_alias(
                    registered_name, "champion"
                )
                champion_run = client.get_run(champion_mv.run_id)
                champion_rmse = champion_run.data.metrics.get("rmse", float("inf"))

                if metrics["rmse"] < champion_rmse:
                    client.set_registered_model_alias(
                        registered_name, "champion", new_version
                    )
                    client.set_registered_model_alias(
                        registered_name, "challenger", champion_mv.version
                    )
                    print(
                        f"[+] [PROMOTION] New champion! Version {new_version} (RMSE: {metrics['rmse']:.4f}) beats v{champion_mv.version} (RMSE: {champion_rmse:.4f})"
                    )
                else:
                    client.set_registered_model_alias(
                        registered_name, "challenger", new_version
                    )
                    print(
                        f"[*] [CHALLENGER] Version {new_version} (RMSE: {metrics['rmse']:.4f}) did not beat champion v{champion_mv.version} (RMSE: {champion_rmse:.4f}). Tagged as @challenger."
                    )
            except Exception:
                client.set_registered_model_alias(
                    registered_name, "champion", new_version
                )
                print(
                    f"[+] [INITIAL CHAMPION] Registered version {new_version} as @champion (RMSE: {metrics['rmse']:.4f})\n"
                )

    return metrics


if __name__ == "__main__":
    train_model()
