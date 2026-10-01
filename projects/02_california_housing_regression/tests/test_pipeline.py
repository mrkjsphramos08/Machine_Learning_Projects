"""
Smoke and data-validation tests for this project.

Standards covered (.agents/rules/mlops_standards.md §4):
  - Smoke tests: every pipeline component must prove it runs end-to-end.
  - Data validation: assert column names, dtypes and non-empty frames BEFORE the
    data reaches the model.

These tests are hermetic. They build a tiny synthetic dataset plus a temporary
config inside tmp_path, so they never read real downloaded data, never overwrite
the project's data/processed files, and never write to the shared mlflow.db.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from src.data import load_config, load_processed_data, prepare_and_save_data
from src.train import build_model, configure_tracking, evaluate_model, train_model

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "bedrooms_ratio",
    "rooms_per_person",
]
TARGET = "MedHouseVal"
N_ROWS = 60
TEST_SIZE = 0.25


@pytest.fixture()
def synthetic_project(tmp_path: Path) -> Path:
    """Create a throwaway raw dataset + config, returning the config path."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    data_dict = {
        feat: [float(i + idx) for i in range(N_ROWS)]
        for idx, feat in enumerate(FEATURES)
    }
    data_dict[TARGET] = [float(i * 1.5) for i in range(N_ROWS)]
    frame = pd.DataFrame(data_dict)
    frame.to_csv(raw_dir / "california_housing.csv", index=False)

    config = {
        "project_name": "02_california_housing_regression",
        "experiment_name": "pytest_california_housing_smoke",
        "run_name": "pytest",
        "mlflow": {
            "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow_test.db').as_posix()}",
            "artifact_root": (tmp_path / "mlruns").as_posix(),
        },
        "paths": {
            "raw_data": (raw_dir / "california_housing.csv").as_posix(),
            "train_data": (
                tmp_path / "data" / "processed" / "train.parquet"
            ).as_posix(),
            "test_data": (tmp_path / "data" / "processed" / "test.parquet").as_posix(),
            "model": (tmp_path / "models" / "model.pkl").as_posix(),
        },
        "dataset": {
            "target_column": TARGET,
            "test_size": TEST_SIZE,
            "random_state": 42,
        },
        "model": {
            "type": "RandomForestRegressor",
            "n_estimators": 5,
            "max_depth": 3,
            "random_state": 42,
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_config_exposes_required_sections(synthetic_project: Path):
    """Config must carry the keys the pipeline relies on."""
    config = load_config(synthetic_project)
    for section in ("experiment_name", "paths", "dataset", "model"):
        assert section in config, f"config is missing the '{section}' section"
    assert Path(config["paths"]["raw_data"]).exists()


def test_prepare_and_save_data_writes_expected_rows(synthetic_project: Path):
    """The prepare stage must produce non-empty train/test Parquet files."""
    prepare_and_save_data(config_path=synthetic_project)
    config = load_config(synthetic_project)

    train_df = pd.read_parquet(config["paths"]["train_data"])
    test_df = pd.read_parquet(config["paths"]["test_data"])

    assert len(train_df) + len(test_df) == N_ROWS
    assert len(test_df) == int(N_ROWS * TEST_SIZE)
    assert not train_df.empty and not test_df.empty


def test_processed_data_validation(synthetic_project: Path):
    """Data validation: expected columns, no null targets, numeric features."""
    prepare_and_save_data(config_path=synthetic_project)
    X_train, X_test, y_train, y_test = load_processed_data(
        config_path=synthetic_project
    )

    assert list(X_train.columns) == FEATURES
    assert list(X_test.columns) == FEATURES
    assert TARGET not in X_train.columns, "target must not leak into the features"
    assert y_train.isna().sum() == 0 and y_test.isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(y_train)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)


def test_configure_tracking_honours_config(synthetic_project: Path):
    """The tracking URI in the config must win over the repo default."""
    config = load_config(synthetic_project)
    assert configure_tracking(config) == config["mlflow"]["tracking_uri"]


def test_evaluate_model_returns_standard_metrics(synthetic_project: Path):
    """Metric keys must stay stable so MLflow history remains comparable."""
    prepare_and_save_data(config_path=synthetic_project)
    X_train, X_test, y_train, y_test = load_processed_data(
        config_path=synthetic_project
    )

    model = build_model(load_config(synthetic_project)["model"])
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics) == {"mse", "rmse", "mae", "r2"}
    assert all(isinstance(value, float) for value in metrics.values())


def test_train_pipeline_smoke(synthetic_project: Path):
    """Smoke test: train_model runs end-to-end and persists the model artifact."""
    prepare_and_save_data(config_path=synthetic_project)

    metrics = train_model(config_path=synthetic_project)
    config = load_config(synthetic_project)

    assert metrics["rmse"] >= 0
    assert Path(config["paths"]["model"]).exists(), "model artifact was not written"
