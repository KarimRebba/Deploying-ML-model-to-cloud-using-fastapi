# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data

# Add the necessary imports for the starter code.
from starter.ml.model import train_model
import joblib

# Load the clean data from the project root.
data_path = Path(__file__).resolve().parents[1] / "data" / "census_clean.csv"
data = pd.read_csv(data_path)


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

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train=X_train, y_train=y_train)
model_path = Path(__file__).resolve().parents[1] / "model" / "rfc_model.joblib"
joblib.dump(model, model_path)
# Save encoder
encoder_path = Path(__file__).resolve().parents[1] / "model" / "encoder.joblib"
joblib.dump(encoder, encoder_path)
# Save label binarizer
lb_path = Path(__file__).resolve().parents[1] / "model" / "label_binarizer.joblib"
joblib.dump(lb, lb_path)
