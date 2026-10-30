"""Unit tests for the FastAPI app."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get_root():
    """
    Test get endpoint by checking the status code and the greeting message
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to my project!"}


def test_post_predict_below_50k():
    """
    Use the the values from the first row in the data to test the inference.
    Check the status code as well as the prediction
    """
    sample = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_predict_above_50k():
    """
    Use the the values from the first row above 50k in the data to test the inference.
    Check the status code as well as the prediction
    """
    sample = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
