import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["greeting"] == "Hello World!"


def test_post_inference_high():
    test_data = {
         "age": 41,
         "workclass": "State-gov",
         "fnlgt": 77516,
         "education": "Doctorate",
         "education-num": 16,
         "marital-status": "Married-civ-spouse",
         "occupation": "Exec-managerial",
         "relationship": "Husband",
         "race": "White",
         "sex": "Male",
         "capital-gain": 5000,
         "capital-loss": 0,
         "hours-per-week": 60,
         "native-country": "United-States"
         }
    response = client.post("/inference/", json=test_data)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_post_inference_low():
    test_data = {
         "age": 41,
         "workclass": "State-gov",
         "fnlgt": 77516,
         "education": "11th",
         "education-num": 7,
         "marital-status": "Married-civ-spouse",
         "occupation": "Handlers-cleaners",
         "relationship": "Husband",
         "race": "White",
         "sex": "Male",
         "capital-gain": 0,
         "capital-loss": 0,
         "hours-per-week": 40,
         "native-country": "United-States"
         }
    response = client.post("/inference/", json=test_data)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"
