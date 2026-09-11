"""
Model training script with MLflow experiment tracking.

================================================================================
MLOps Tracking Pattern — Core MLflow Calls Explained:
================================================================================
1. mlflow.set_experiment(name):
   - Organizes individual runs under a named project/experiment banner.
   - Without this, runs are lumped into a generic "Default" experiment.

2. mlflow.start_run(run_name=...):
   - Opens an active tracking context (a single trial/execution).
   - Automatically captures system metadata (start time, user, git commit hash).
   - Using `with mlflow.start_run():` guarantees the run is properly marked 
     as FINISHED or FAILED even if an error occurs.

3. mlflow.log_param(key, value) / mlflow.log_params(dict):
   - Logs immutable INPUTS and configuration (hyperparameters, split sizes, random seeds).
   - Essential for answering: "What exact settings produced this specific model?"

4. mlflow.log_metric(key, value) / mlflow.log_metrics(dict):
   - Logs numeric OUTPUTS and evaluation performance (MSE, RMSE, R2, accuracy).
   - Allows plotting curves over time and comparing models side-by-side in the UI.

5. mlflow.sklearn.log_model(sk_model, artifact_path):
   - Serializes and stores the actual trained model artifact along with a Conda/pip
     environment file (MLmodel specification).
   - Enables reproducible loading later (`mlflow.sklearn.load_model(...)`) for 
     serving, batch inference, or promotion to a Model Registry.
================================================================================
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn

from src.data import prepare_data


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model():
    # 1. Load configuration (keep code decoupled from hardcoded hyperparameters)
    config = load_config()

    experiment_name = config.get("experiment_name", "default_experiment")
    run_name = config.get("run_name", "run")
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})

    # 2. Tell MLflow which experiment group to track this under
    mlflow.set_experiment(experiment_name)

    # 3. Start tracked execution run
    with mlflow.start_run(run_name=run_name):
        print(f"[*] Started MLflow run: '{run_name}' under experiment '{experiment_name}'")

        # 4. Prepare data
        test_size = dataset_cfg.get("test_size", 0.2)
        random_state = dataset_cfg.get("random_state", 42)
        X_train, X_test, y_train, y_test = prepare_data(
            test_size=test_size, random_state=random_state
        )

        # 5. Log Parameters (Inputs)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("data_random_state", random_state)
        mlflow.log_param("model_type", model_cfg.get("type", "RandomForestRegressor"))
        mlflow.log_param("n_estimators", model_cfg.get("n_estimators", 100))
        mlflow.log_param("max_depth", model_cfg.get("max_depth", 6))
        mlflow.log_param("model_random_state", model_cfg.get("random_state", 42))

        # 6. Train model
        model = RandomForestRegressor(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 6),
            random_state=model_cfg.get("random_state", 42),
        )
        model.fit(X_train, y_train)

        # 7. Evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # 8. Log Metrics (Outputs)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # 9. Log Model Artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=None,
        )

        print("[+] Training completed successfully!")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE:  {mae:.4f}")
        print(f"    R2:   {r2:.4f}")
        print("[+] Run metrics and model artifact logged to MLflow.\n")


if __name__ == "__main__":
    train_model()
