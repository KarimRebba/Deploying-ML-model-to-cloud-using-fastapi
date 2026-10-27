# Put the code for your API here.
import joblib

model = joblib.load("model/random_forest.joblib")
encoder = joblib.load("model/encoder.joblib")
lb = joblib.load("model/label_binarizer.joblib")
