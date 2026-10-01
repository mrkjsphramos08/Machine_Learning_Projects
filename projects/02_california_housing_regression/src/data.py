"""
Data ingestion and preparation.

MLOps pattern: ingestion/cleaning lives here, model fitting lives in train.py.
Keeping them in separate modules makes the pipeline DAG explicit so DVC can
cache each stage independently.

Run as a DVC stage:   python -m src.data
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.paths import PROJECT_ROOT, ensure_parent, resolve

DEFAULT_CONFIG_PATH = "configs/config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the project YAML configuration.

    Relative config paths are resolved against the project root.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_data_path: str | Path) -> pd.DataFrame:
    """Read the immutable raw dataset. This file is versioned by DVC."""
    return pd.read_csv(resolve(str(raw_data_path)))


def clean_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Apply project-specific cleaning rules.

    Default implementation drops rows with a missing target, which cannot be
    used for supervised training. Replace with real cleaning as you learn the
    dataset (duplicates, outliers, dtypes, invalid categories).
    """
    cleaned = df.dropna(subset=[target_column]).reset_index(drop=True)
    dropped = len(df) - len(cleaned)
    if dropped:
        print(f"[!] Dropped {dropped} row(s) with a missing target.")
    return cleaned


def prepare_and_save_data(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    """Split the raw data into train/test sets and persist them as Parquet.

    Parquet is preferred over CSV for pipeline intermediates: it is typed,
    columnar, compressed, and much faster to reload.
    """
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})
    target_column = dataset_cfg.get("target_column", "target")

    print(f"[*] Loading raw dataset: {paths_cfg['raw_data']}")
    df = clean_data(load_raw_data(paths_cfg["raw_data"]), target_column)
    df = engineer_features_california(df)

    train_df, test_df = train_test_split(
        df,
        test_size=dataset_cfg.get("test_size", 0.2),
        random_state=dataset_cfg.get("random_state", 42),
    )

    train_path = ensure_parent(
        resolve(paths_cfg.get("train_data", "data/processed/train.parquet"))
    )
    test_path = ensure_parent(
        resolve(paths_cfg.get("test_data", "data/processed/test.parquet"))
    )
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"[+] Wrote processed data to {train_path.parent}")
    print(f"    train: {len(train_df)} rows | test: {len(test_df)} rows")


def load_processed_data(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the processed Parquet files and split them into features and target."""
    config = load_config(config_path)
    target_column = config.get("dataset", {}).get("target_column", "target")
    paths_cfg = config.get("paths", {})

    train_df = pd.read_parquet(
        resolve(paths_cfg.get("train_data", "data/processed/train.parquet"))
    )
    test_df = pd.read_parquet(
        resolve(paths_cfg.get("test_data", "data/processed/test.parquet"))
    )

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    return X_train, X_test, y_train, y_test


def engineer_features_california(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to the California housing dataset."""
    print("[*] Applying feature engineering...")

    # Feature engineering pipeline
    df = df.copy()

    # Create derived features
    # 1. Bedrooms ratio: ratio of bedrooms to rooms per household
    df["bedrooms_ratio"] = df["AveBedrms"] / df["AveRooms"]

    # 2. Rooms per person: average number of rooms per person in a household
    df["rooms_per_person"] = df["AveRooms"] / df["AveOccup"]

    return df


if __name__ == "__main__":
    prepare_and_save_data()
