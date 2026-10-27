import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference,
)
from starter.ml.data import process_data


def test_process_data():
    data = pd.DataFrame({
        "age": [25, 30, 40],
        "workclass": ["Private", "Government", "Private"],
        "salary": ["<=50K", ">50K", "<=50K"],
    })

    X, y, encoder, lb = process_data(
        data,
        categorical_features=["workclass"],
        label="salary",
        training=True,
    )

    assert X.shape == (3, 3)
    assert len(y) == 3
    assert encoder is not None
    assert lb is not None


def test_train_model():
    X = np.array([
        [20, 0],
        [30, 1],
        [40, 0],
        [50, 1],
    ])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics():
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert np.isclose(recall, 2 / 3)
    assert np.isclose(fbeta, 0.8)

def test_inference():
    X = np.array([
        [20, 0],
        [30, 1],
        [40, 0],
        [50, 1],
    ])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(y)


