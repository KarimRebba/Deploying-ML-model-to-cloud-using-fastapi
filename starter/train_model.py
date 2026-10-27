# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data

# Add the necessary imports for the starter code.
from starter.ml.model import (
    train_model,
    compute_slice_metrics,
    inference,
    compute_model_metrics
)
import joblib

# Load the clean data from the project root.

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATA_PATH = ROOT / "data" / "census_clean.csv"
data = pd.read_csv(DATA_PATH)


# Optional: use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
    training=False, encoder=encoder, lb=lb,
)

# Train model and evaluate model
model = train_model(X_train, y_train)
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Test precision={precision:.4f}, recall={recall:.4f}, "
      f"fbeta={fbeta:.4f}")

compute_slice_metrics(model, test, cat_features, "salary", encoder, lb)

# Save artifacts
joblib.dump(encoder, MODEL_DIR / "encoder.joblib")
joblib.dump(lb, MODEL_DIR / "label_binarizer.joblib")
joblib.dump(model, MODEL_DIR / "rfc_model.joblib")