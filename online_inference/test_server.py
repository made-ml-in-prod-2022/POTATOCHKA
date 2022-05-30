from fastapi.testclient import TestClient
from server import app
from config import COLUMNS
import numpy as np

client = TestClient(app)


def test_main_part():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Start online model inference"


def test_predict_positive_request():
    with client:
        data = [0, 66, 0, 3, 178, 228, 1, 0, 165, 1, 1.0, 1, 2, 2]
        request = {"data": [data], "features": COLUMNS}
        response = client.get(
            "/predict/", json=request
        )
        assert response.status_code == 200
        answer = response.json()
        assert np.isclose(answer["probability"][0], 0.9, atol=0.1)


def test_predict_negative_request():
    with client:
        data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        request = {"data": [data], "features": COLUMNS}
        response = client.get(
            "/predict/", json=request
        )
        assert response.status_code == 200
        answer = response.json()
        assert np.isclose(answer["probability"][0], 0.1, atol=0.1)


def test_predict_on_empty_data():
    with client:
        data = []
        request = {"data": [data], "features": COLUMNS}
        response = client.get(
            "/predict/", json=request
        )
        assert response.status_code == 422
        answer = response.json()
        assert answer["detail"][0]["msg"] == "ensure this value has at least 14 items"


def test_predict_different_feats():
    with client:
        data = [0, 66, 0, 3, 178, 228, 1, 0, 165, 1, 1.0, 1, 2, 2]
        request = {"data": [data], "features": COLUMNS[::-1]}
        response = client.get(
            "/predict/", json=request
        )
        assert response.status_code == 400
        answer = response.json()
        assert answer["detail"] == "mismatch in columns"


def test_predict_with_none():
    with client:
        data = [0, 66, 0, 3, np.nan, 228, 1, 0, 165, 1, 1.0, 1, 2, 2]
        request = {"data": [data], "features": COLUMNS}
        response = client.get(
            "/predict/", json=request
        )
        assert response.status_code == 400
        answer = response.json()
        assert answer["detail"] == "Nans in input"


def test_predict_with_str():
    with client:
        data = [0, 66, 0, 3, "kek", 228, 1, 0, 165, 1, 1.0, 1, 2, 2]
        request = {"data": [data], "features": COLUMNS}
        response = client.get(
            "/predict/", json=request
        )
        assert response.status_code == 422
        answer = response.json()
        assert answer["detail"][0]["msg"] == "value is not a valid float"


def test_predict_with_incorrect_binary_val():
    with client:
        data = [1] * len(COLUMNS)
        data[2] = 1000
        request = {"data": [data], "features": COLUMNS}
        response = client.get(
            "/predict/", json=request
        )
        assert response.status_code == 400
        answer = response.json()
        assert answer["detail"] == "incorrect value 1000.0 in sex"


def test_predict_with_incorrect_categorical_val():
    with client:
        data = [0.5] * len(COLUMNS)
        data[3] = 1000
        request = {"data": [data], "features": COLUMNS}
        response = client.get(
            "/predict/", json=request
        )
        assert response.status_code == 400
        answer = response.json()
        assert answer["detail"] == "incorrect value 1000.0 in cp"
