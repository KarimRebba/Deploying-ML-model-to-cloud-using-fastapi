"""Send one POST request to the live API and print the result."""
import requests

URL = "https://deploying-ml-model-to-cloud-using-fastapi.onrender.com/predict"

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

response = requests.post(URL, json=sample, timeout=80)
print(f"Status code: {response.status_code}")
print(f"Result: {response.json()}")
