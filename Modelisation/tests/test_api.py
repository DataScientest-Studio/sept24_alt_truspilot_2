from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "database_available" in data


def test_model_info_returns_200():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "TF-IDF + Logistic Regression"
    assert data["problem_type"] == "binary classification"


def test_predict_returns_prediction():
    response = client.post(
        "/predict",
        json={"text": "This product is amazing and I love it"}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["text"] == "This product is amazing and I love it"
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["negative", "positive"]
    assert "probability_negative" in data
    assert "probability_positive" in data


def test_predict_empty_text_returns_422():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422