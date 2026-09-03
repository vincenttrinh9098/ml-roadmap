"""
app.py

Stage 9, Project 1: Wrap your fraud model in a real HTTP API.

Run this with: uvicorn app:app --reload
Then visit http://127.0.0.1:8000/docs -- FastAPI auto-generates an interactive API
explorer from your code, letting you test the /predict endpoint directly in the browser.

Syntax note on "uvicorn app:app": the first "app" is this file (app.py), the second "app"
is the FastAPI object defined inside it (see below). --reload auto-restarts the server
whenever you save changes to this file, useful during development.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger("fraud-api")

# load the serialized model and scaler
# model = joblib.load("fraud_model.joblib")
# scaler = joblib.load("scaler.joblib")
# Loading happens ONCE, at server startup -- not per-request. This is a big part of why
# a served model is fast: the expensive part (loading weights into memory) happens once,
# and every request afterward just runs inference on an already-loaded model.

model = joblib.load("fraud_model.joblib")
scaler = joblib.load("scaler.joblib")
app = FastAPI(title="Fraud Detection API")

# Syntax note: pydantic's BaseModel defines the exact shape of data your API expects.
# FastAPI uses this to automatically validate incoming requests -- if a request is missing
# a field, or sends a string where a float is expected, FastAPI rejects it with a clear
# error BEFORE your code ever runs, rather than crashing partway through prediction.
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


THRESHOLD = 0.6  # your chosen threshold from Stage 6 

@app.get("/health")
def health():
    """A basic liveness check -- confirms the API is running at all, no model logic involved.
    Real production systems check this endpoint constantly (every few seconds) to detect
    if a deployed service has crashed."""
    # TODO: return something like {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    logger.info(f"Received prediction request, Amount={transaction.Amount}")

    # 1. Convert Pydantic object → dictionary
    transaction_dict = transaction.model_dump()

    # 2. Convert dictionary → 2D NumPy array
    # Make sure this order EXACTLY matches your training features
    feature_order = [
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount"
    ]

    transaction_array = np.array(
        [[transaction_dict[col] for col in feature_order]],
        dtype=float
    )

    # 3. Scale the SAME features that were scaled during training
    # This assumes your scaler was trained specifically on Time and Amount.
    features_to_scale = ["Time", "Amount"]

    scale_indices = [feature_order.index(col) for col in features_to_scale]

    transaction_array[:, scale_indices] = scaler.transform(
        transaction_array[:, scale_indices]
    )

    # 4. Get fraud probability
    y_pred_prob = model.predict_proba(transaction_array)[:, 1]

    # 5. Apply your threshold
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

    logger.info(f"Prediction complete: fraud_probability={y_pred_prob[0]:.4f}, is_fraud={bool(y_pred[0])}")

    # 6. Return JSON-friendly values
    return {
        "fraud_probability": float(y_pred_prob[0]),
        "is_fraud": bool(y_pred[0]),
        "threshold_used": THRESHOLD
    }