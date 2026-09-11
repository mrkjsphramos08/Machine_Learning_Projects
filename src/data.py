"""
Data loading and preprocessing module.

In MLOps, separating data ingestion and preparation into modular functions ensures:
1. Reusability across training, evaluation, and inference pipelines.
2. Clear testability (we can unit test data shapes and null-value handling).
3. Seamless integration with data versioning tools like DVC later on.
"""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def load_raw_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Loads the raw dataset. Using scikit-learn's built-in diabetes dataset."""
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target
    return X, y


def prepare_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits raw data into train and test sets."""
    X, y = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
