"""
Smoke tests for the MLOps pipeline.

In MLOps CI/CD, smoke tests answer: 'Did recent code changes break the pipeline?'
Rather than testing model accuracy, smoke tests ensure scripts run end-to-end without crashing.
"""

from src.data import prepare_data
from src.train import train_model


def test_data_loader():
    """Verify data loading and splitting returns expected shapes."""
    X_train, X_test, y_train, y_test = prepare_data(test_size=0.2, random_state=42)
    assert len(X_train) > 0, "Train set should not be empty"
    assert len(X_test) > 0, "Test set should not be empty"
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_train_pipeline_smoke():
    """Smoke test: Ensure train_model runs end-to-end without unhandled exceptions."""
    try:
        train_model()
    except Exception as e:
        assert False, f"train_model failed with error: {e}"
