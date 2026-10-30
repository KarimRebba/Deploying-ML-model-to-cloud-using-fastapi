from fastapi import FastAPI
from starter.ml.model import inference, load_model
from starter.ml.data import process_data
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd

# Define Paths
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

# Load artifacts for inference
model = load_model(MODEL_DIR / "rfc_model.joblib")
encoder = load_model(MODEL_DIR / "encoder.joblib")
lb = load_model(MODEL_DIR / "label_binarizer.joblib")

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

# Create app
app = FastAPI(
    title="Income prediction API",
    description="Predicts whether annual income is above or below 50k",
)


class CensusData(BaseModel):
    """Request body; aliases map to the original hyphenated CSV columns."""

    # Provide example values in the pydantic model
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
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
            ]
        },
    )

    # Field order matches the CSV column order
    age: int = Field(examples=[39])
    workclass: str = Field(examples=["State-gov"])
    fnlgt: int = Field(examples=[77516])
    education: str = Field(examples=["Bachelors"])
    education_num: int = Field(alias="education-num", examples=[13])
    marital_status: str = Field(alias="marital-status", examples=["Never-married"])
    occupation: str = Field(examples=["Adm-clerical"])
    relationship: str = Field(examples=["Not-in-family"])
    race: str = Field(examples=["White"])
    sex: str = Field(examples=["Male"])
    capital_gain: int = Field(alias="capital-gain", examples=[2174])
    capital_loss: int = Field(alias="capital-loss", examples=[0])
    hours_per_week: int = Field(alias="hours-per-week", examples=[40])
    native_country: str = Field(alias="native-country", examples=["United-States"])


# Get request that returns a welcome message
@app.get("/")
async def welcome_message() -> dict[str, str]:
    return {"greeting": "Welcome to my project!"}


# Post request that performs model inference
@app.post("/predict")
async def predict(data: CensusData) -> dict[str, str]:
    df = pd.DataFrame([data.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    pred = inference(model, X)
    return {"prediction": lb.inverse_transform(pred)[0]}
