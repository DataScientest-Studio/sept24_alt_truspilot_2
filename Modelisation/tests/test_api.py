from fastapi.testclient import TestClient

import api
from api import app, API_KEY


client = TestClient(app)

AUTH_HEADERS = {
    "X-API-Key": API_KEY
}


class FakeModel:
    def predict(self, texts):
        return [1]

    def predict_proba(self, texts):
        return [[0.10, 0.90]]


def test_root_returns_200():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    assert "database_available" in data


def test_model_info_returns_200():
    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()

    assert data["problem_type"] == "binary classification"
    assert "labels" in data


def test_predict_returns_prediction(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())

    response = client.post(
        "/predict",
        headers=AUTH_HEADERS,
        json={"text": "This product is amazing and I love it"}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["text"] == "This product is amazing and I love it"
    assert data["prediction"] == 1
    assert data["label"] == "positive"
    assert data["probability_negative"] == 0.10
    assert data["probability_positive"] == 0.90


def test_predict_without_api_key_returns_401():
    response = client.post(
        "/predict",
        json={"text": "This product is amazing and I love it"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_predict_empty_text_returns_422():
    response = client.post(
        "/predict",
        headers=AUTH_HEADERS,
        json={"text": ""}
    )

    assert response.status_code == 422