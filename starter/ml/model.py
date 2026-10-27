from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data
from pathlib import Path

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : rfc
        Trained machine learning model.
    """
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    return rfc

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def compute_slice_metrics(model, df, categorical_features, label,
                        encoder, lb, output_path=
                        Path(__file__).resolve().parents[1] /"slice_output.txt"):
    """Compute precision, recall and F-beta for every value of each
    categorical feature and write the results in a file.
    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    df : pd.DataFrame
        Dataframe containing the features and label
    categorical_features: list[str]
        List containing the names of the categorical features
    label : str
        Name of the label column in `data`
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer
    output_path: str
        Path to the .txt file containing the results
    Returns
    -------
    results : list of dict
    """
    results = []
    for feature in categorical_features:
        for value in df[feature].unique():
            slice_df = df[df[feature] == value]
            X_slice, y_slice, _, _ = process_data(
                slice_df,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            results.append({
                "feature": feature,
                "value": value,
                "n": len(slice_df),
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            })

    with open(output_path, "w") as f:
        for result in results:
            f.write(
                f"{result['feature']}={result['value']} (n={result['n']}): "
                f"precision={result['precision']:.4f}, "
                f"recall={result['recall']:.4f}, "
                f"fbeta={result['fbeta']:.4f}\n"
            )
    return results