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
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    model_type = model_cfg.get("type", "RandomForestClassifier")
    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 6),
            random_state=model_cfg.get("random_state", 42),
        )
    elif model_type == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            max_iter=model_cfg.get("max_iter", 100),
            learning_rate=model_cfg.get("learning_rate", 0.05),
            max_depth=model_cfg.get("max_depth", 4),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 20),
            l2_regularization=model_cfg.get("l2_regularization", 0.0),
            random_state=model_cfg.get("random_state", 42),
        )
    elif model_type == "LogisticRegression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=model_cfg.get("C", 1.0),
                        random_state=model_cfg.get("random_state", 42),
                        max_iter=model_cfg.get("max_iter", 1000),
                    ),
                ),
            ]
        )
    elif model_type == "SVC":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=model_cfg.get("C", 1.0),
                        kernel=model_cfg.get("kernel", "rbf"),
                        probability=True,
                        random_state=model_cfg.get("random_state", 42),
                    ),
                ),
            ]
        )
    elif model_type == "KNeighborsClassifier":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=model_cfg.get("n_neighbors", 5),
                        weights=model_cfg.get("weights", "uniform"),
                    ),
                ),
            ]
        )
    elif model_type == "MLPClassifier":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=tuple(
                            model_cfg.get("hidden_layer_sizes", [32, 16])
                        ),
                        max_iter=model_cfg.get("max_iter", 500),
                        random_state=model_cfg.get("random_state", 42),
                    ),
                ),
            ]
        )
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. "
            "Supported: 'RandomForestClassifier', 'HistGradientBoostingClassifier', "
            "'LogisticRegression', 'SVC', 'KNeighborsClassifier', 'MLPClassifier'."
        )


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Compute classification metrics on the held-out test set."""
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    return metrics


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
            "model_type": model_cfg.get("type", "RandomForestClassifier"),
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

        registered_name = config.get("registered_model_name")
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        input_example = X_test.head(3)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_name,
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        print("[+] Training completed successfully!")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-score:  {metrics['f1']:.4f}")
        if "roc_auc" in metrics:
            print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
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
                champion_score = champion_run.data.metrics.get("roc_auc", 0.0)

                if metrics["roc_auc"] > champion_score:
                    client.set_registered_model_alias(
                        registered_name, "champion", new_version
                    )
                    client.set_registered_model_alias(
                        registered_name, "challenger", champion_mv.version
                    )
                    print(
                        f"[+] [PROMOTION] New champion! Version {new_version} (ROC-AUC: {metrics['roc_auc']:.4f}) beats v{champion_mv.version} (ROC-AUC: {champion_score:.4f})"
                    )
                else:
                    client.set_registered_model_alias(
                        registered_name, "challenger", new_version
                    )
                    print(
                        f"[*] [CHALLENGER] Version {new_version} (ROC-AUC: {metrics['roc_auc']:.4f}) did not beat champion v{champion_mv.version} (ROC-AUC: {champion_score:.4f}). Tagged as @challenger."
                    )
            except Exception:
                # No champion exists yet -> tag first version as initial champion
                client.set_registered_model_alias(
                    registered_name, "champion", new_version
                )
                print(
                    f"[+] [INITIAL CHAMPION] Version {new_version} tagged as @champion."
                )

    return metrics


if __name__ == "__main__":
    train_model()
