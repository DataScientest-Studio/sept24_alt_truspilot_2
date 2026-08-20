from fastapi.testclient import TestClient

from api import app, API_KEY


client = TestClient(app)

AUTH_HEADERS = {
    "X-API-Key": API_KEY
}


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


def test_predict_returns_prediction():
    response = client.post(
        "/predict",
        headers=AUTH_HEADERS,
        json={"text": "This product is amazing and I love it"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "text" in data
    assert "prediction" in data
    assert "label" in data
    assert "probability_negative" in data
    assert "probability_positive" in data
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["positive", "negative"]


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