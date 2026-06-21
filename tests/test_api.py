import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def make_mock_model(prob: float = 0.15):
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - prob, prob]])
    return mock


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the model pre-loaded as a mock."""
    mock_wrapper = MagicMock()
    mock_wrapper.version = "test-v1"
    mock_wrapper.name = "MockXGBClassifier"
    mock_wrapper.predict_proba.return_value = np.array([0.15])

    with patch("api.model_loader.load_model", return_value=mock_wrapper):
        import api.app as app_module
        from api.app import app

        app_module._model = mock_wrapper  # force inject
        yield TestClient(app)


VALID_PAYLOAD = {
    "AMT_CREDIT": 540000.0,
    "AMT_INCOME_TOTAL": 135000.0,
    "AMT_ANNUITY": 26000.0,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -3000,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_response_has_required_fields(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert "default_probability" in data
        assert "risk_label" in data
        assert "risk_score" in data
        assert "model_version" in data

    def test_predict_probability_in_range(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        prob = resp.json()["default_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_label_valid(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        label = resp.json()["risk_label"]
        assert label in ["LOW", "MEDIUM", "HIGH"]

    def test_predict_risk_score_in_range(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        score = resp.json()["risk_score"]
        assert 0 <= score <= 1000

    def test_predict_rejects_positive_days_birth(self, client):
        bad_payload = {**VALID_PAYLOAD, "DAYS_BIRTH": 100}  # must be negative
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422  # Pydantic validation error

    def test_predict_rejects_zero_credit(self, client):
        bad_payload = {**VALID_PAYLOAD, "AMT_CREDIT": 0}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_predict_rejects_missing_required_fields(self, client):
        resp = client.post("/predict", json={"AMT_CREDIT": 500000})
        assert resp.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        resp = client.post("/predict/batch", json={"applications": [VALID_PAYLOAD]})
        assert resp.status_code == 200

    def test_batch_predict_count_matches(self, client):
        resp = client.post("/predict/batch", json={"applications": [VALID_PAYLOAD, VALID_PAYLOAD]})
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_predict_avg_probability_in_range(self, client):
        resp = client.post("/predict/batch", json={"applications": [VALID_PAYLOAD]})
        avg = resp.json()["avg_default_probability"]
        assert 0.0 <= avg <= 1.0

    def test_batch_rejects_empty_list(self, client):
        resp = client.post("/predict/batch", json={"applications": []})
        assert resp.status_code == 422


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_has_name(self, client):
        resp = client.get("/model/info")
        assert "model_name" in resp.json()
