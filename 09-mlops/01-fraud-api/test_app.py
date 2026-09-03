"""
test_app.py

Stage 9: automated tests for the fraud detection API.

**New concept: FastAPI's TestClient.** Up to now, you've tested this API by actually running
`uvicorn` in one terminal and sending real HTTP requests from another. TestClient does something
different: it runs your FastAPI app directly in-process, in memory, simulating requests without
ever opening a real network socket or needing a separate server running at all.

Why this matters: these tests can run in a fraction of a second, work identically on your laptop
or in an automated CI pipeline (like GitHub Actions) with zero manual setup, and don't depend on
any port being free or a server already running. This is the actual mechanism that makes
"automated testing" automated -- no human has to remember to start uvicorn first.

Run this with: python3 -m pytest test_app.py -v
(-v = verbose, shows each individual test's name and pass/fail, not just a summary count)
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)



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
        "V24": -10.1,
        "V25": 0.3,
        "V26": -0.2,
        "V27": 1.1,
        "V28": 0.5,
        "Amount": 1.00
}


incomplete_transaction = {

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
        "V23": -0.7, # missing 24
        "V25": 0.3,
        "V26": -0.2,
        "V27": 1.1,
        "V28": 0.5,
        "Amount": 1.00
}


improper_transaction = {

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
        "V24":-1.1,
        "V25": 0.3,
        "V26": -0.2,
        "V27": 1.1,
        "V28": 0.5,
        "Amount": "not_a_float"
}


negative_transaction = {

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
        "V24": -500000.1,
        "V25": 0.3,
        "V26": -0.2,
        "V27": 1.1,
        "V28": 0.5,
        "Amount": 1.00
}
def test_health_check():
    """The simplest possible test: does the API respond at all, with the expected shape?"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_transaction_returns_200():
    """A well-formed request with all 30 fields should succeed."""


    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200


def test_predict_response_has_expected_keys():
    """Beyond just succeeding, does the response actually contain what callers of this
    API would depend on? This is the kind of test that catches an accidental typo in a
    key name (e.g. 'fraud_probablity') before it breaks something that depends on this API."""

    response = client.post("/predict", json=sample_transaction)
    assert "fraud_probability" in response.json()
    assert "is_fraud" in response.json()
    assert "threshold_used" in response.json()
    


def test_predict_probability_is_valid_range():
    """A probability must be between 0 and 1 -- this should ALWAYS be true regardless of
    input, so it's a good candidate for a dedicated test. Guiding question: what would it
    mean if this test ever failed? What kind of bug would produce a probability outside
    this range?"""
    response = client.post("/predict", json=sample_transaction)
    assert 0.0 <= response.json()["fraud_probability"] <= 1.0


def test_predict_missing_field_returns_422():
    """This is the test version of what you manually checked earlier by deleting a field
    from test_api.py by hand. Automating it means this protection can never silently break
    without you noticing -- if someone edits the Transaction model incorrectly later, this
    test catches it immediately instead of relying on someone remembering to test by hand."""
    response = client.post("/predict", json=incomplete_transaction)
    assert response.status_code == 422



def test_predict_wrong_type_returns_422():
    """Similar idea, different failure mode: what happens if a field has the WRONG TYPE
    (a string where a float is expected), rather than being missing entirely?"""
    response = client.post("/predict", json=improper_transaction)
    assert response.status_code == 422


def test_known_high_risk_transaction_flags_as_fraud():
    """A regression test: this specific transaction (very negative V14, among other
    extreme values) should reliably be flagged as fraud, given your trained model.
    If this test ever starts failing after a future model change, that's a real signal
    worth investigating -- did the new model get WORSE, or did its behavior legitimately
    change for a defensible reason?"""
    response = client.post("/predict", json=sample_transaction)
    assert response.json()["is_fraud"] == True

