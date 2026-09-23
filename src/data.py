"""
Data loading and preprocessing module.

In MLOps, separating data ingestion and preparation into modular functions ensures:
1. Reusability across training, evaluation, and inference pipelines.
2. Clear testability (we can unit test data shapes and null-value handling).
3. Seamless integration with data versioning tools like DVC later on.
"""

import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_and_save_data(
    raw_data_path: str = "data/raw/data.csv",
    processed_dir: str = "data/processed",
    config_path: str = "configs/train_config.yaml",
) -> None:
    """Reads raw CSV, splits into train/test sets, and saves as Parquet files."""
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    test_size = dataset_cfg.get("test_size", 0.2)
    random_state = dataset_cfg.get("random_state", 42)

    print(f"[*] Loading raw dataset from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, "train.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"[+] Successfully generated processed datasets:")
    print(f"    Train: {train_path} ({len(train_df)} rows)")
    print(f"    Test:  {test_path} ({len(test_df)} rows)")


def load_processed_data(
    processed_dir: str = "data/processed",
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads processed parquet files into feature and target splits."""
    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_and_save_data()
