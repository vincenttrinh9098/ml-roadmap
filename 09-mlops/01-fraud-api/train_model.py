"""
train_model.py

Stage 9, Project 1: Serialize your Stage 6 fraud model so an API can load and use it
without retraining every time.

This retrains a simplified version of your Stage 6 fraud detection pipeline (raw XGBoost,
no scale_pos_weight/SMOTE, matching the winning approach from that comparison), then saves
both the trained model AND the fitted scaler to disk.

Syntax note: `joblib.dump(obj, path)` serializes a Python object to a file, and
`joblib.load(path)` reads it back. This is the standard serialization tool for scikit-learn-style
objects (including XGBoost models) -- it's what lets a completely separate process (your API)
use a model without ever running the training code again.

Run this once with: python3 train_model.py
It should produce two files: fraud_model.joblib and scaler.joblib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"

#       load the dataset from URL
#       split X (all columns except 'Class') / y ('Class'), stratified train/test split
#       (this script only needs train/test, not train/val/test, since you're not comparing
#       approaches here -- you already did that comparison in Stage 6)

data = pd.read_csv(URL)
X = data.drop(columns=["Class"])
y = data["Class"]
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=.30,
    random_state=42,
    stratify=y,

)

#       fit a StandardScaler on ONLY the 'Amount' and 'Time' columns of X_train
#       (V1-V28 are already PCA-transformed/scaled, same reasoning as Stage 6)
#       transform Amount/Time in both X_train and X_test using that fitted scaler

scaler = StandardScaler()
features_to_scale = ["Time", "Amount"]

X_train[features_to_scale] = scaler.fit_transform(
    X_train[features_to_scale]
)
X_test[features_to_scale] = scaler.transform(
    X_test[features_to_scale]
)

#       train XGBClassifier() with the same hyperparameters as your Stage 6 winning model
#       (n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42 -- check your
#       Stage 6 notebook for the exact values you used)

model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print(y_pred_prob)

#       quick sanity check -- print accuracy or a confusion matrix on X_test, just to confirm
#       this retrained model performs similarly to what you got in Stage 6 (it won't be
#       identical since this is a fresh train/test split, but it should be in the same ballpark)




precision_xgb = precision_score(y_test, y_pred)
recall_xgb = recall_score(y_test, y_pred)
f1_xgb = f1_score(y_test, y_pred)
roc_auc_xgb = roc_auc_score(y_test, y_pred_prob)
avg_precision_xgb = average_precision_score(y_test, y_pred_prob)
print(confusion_matrix(y_test, y_pred))
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1:", f1_xgb)
print("ROC-AUC:", roc_auc_xgb)
print("Avg Precision:", avg_precision_xgb)


# save both the trained model and the fitted scaler:
joblib.dump(model, "fraud_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Done. Check for fraud_model.joblib and scaler.joblib in this folder.")
