import requests

API_URL = "https://udacity-fastapi-project-05ka.onrender.com/inference/"

post_data = {
    "age": 42,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5000,
    "capital-loss": 0,
    "hours-per-week": 35,
    "native-country": "Germany",
}

response = requests.post(API_URL, json=post_data)

print(f"Status code: {response.status_code}")
print(f"Inference result: {response.json()}")
