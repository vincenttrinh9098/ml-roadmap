"""
test_api.py

Stage 9, Project 1: Actually call your running API, like a real client application would.

Prerequisite: app.py must already be running in another terminal
(uvicorn app:app --reload), since this script sends a real HTTP request to it.

Run this with: python3 test_api.py
"""

import requests

URL = "http://127.0.0.1:8000/predict"

# build a sample transaction dict with all 30 fields (Time, V1-V28, Amount).
# You can pull one real row directly from the dataset for a realistic test -- e.g., grab
# the first row of X_test from train_model.py and print it as a dict to copy the values here.
# Using an all-zeros dummy transaction will technically work but won't tell you much about
# whether the model's actual behavior is correct.

sample_transaction = {

    "Time": 132345,
    "V1": -5.2,
    "V2": 5.1,
    "V3": -7.8,
    "V4": 6.3,
    "V5": -4.7,
    "V6": -2.1,
    "V7": -8.4,
    "V8": 2.9,
    "V9": -4.2,
    "V10": -8.7,
    "V11": 7.1,
    "V12": -9.3,
    "V13": 0.4,
    "V14": -10.2,
    "V15": 1.2,
    "V16": -6.8,
    "V17": -12.5,
    "V18": -5.4,
    "V19": 2.1,
    "V20": 1.8,
    "V21": 1.5,
    "V22": 0.2,
    "V23": -0.7,
    "V24": -1.1,
    "V25": 0.3,
    "V26": -0.2,
    "V27": 1.1,
    "V28": 0.5,
    "Amount": 1.00
}

response = requests.post(URL, json=sample_transaction)
print(response.status_code)
print(response.json())


