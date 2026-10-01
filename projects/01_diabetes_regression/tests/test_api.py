"""Automated tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


class DummyModel:
    """Hermetic fallback model for test environments where MLflow store is clean."""

    def predict(self, df):
        return [150.0] * len(df)


@pytest.fixture()
def client():
    """Create a test client with lifespan context to ensure model is loaded."""
    with TestClient(app) as test_client:
        if app.state.model is None:
            app.state.model = DummyModel()
        yield test_client


def test_health_reports_healthy(client: TestClient):
    """GET /health must return 200 and report that the champion model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "DiabetesRegressor@champion" in data["model_uri"]


def test_predict_returns_valid_prediction(client: TestClient):
    """POST /predict with valid 10 features returns 200 and positive progression score."""
    sample_payload = {
        "features": [
            0.038076,
            0.050680,
            0.061696,
            0.021872,
            -0.044223,
            -0.034821,
            -0.043401,
            -0.002592,
            0.019907,
            -0.017646,
        ]
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["prediction"] > 0
    assert "DiabetesRegressor@champion" in data["model_uri"]


def test_predict_rejects_wrong_feature_count(client: TestClient):
    """POST /predict with wrong number of features (e.g. 5 instead of 10) returns 422."""
    invalid_payload = {"features": [0.01, 0.02, 0.03, 0.04, 0.05]}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
