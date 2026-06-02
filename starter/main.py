# Put the code for your API here.

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
import pickle
from pathlib import Path
import yaml
from starter.ml.model import inference
from starter.ml.data import process_data

# Instantiate the app.
app = FastAPI(
    title="Model inference API",
    description="API for model inference of a classification model using publicly available Census Bureau data",
    version="1.0.0",
)


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


class CensusData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 41,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Doctorate",
                "education-num": 16,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 5000,
                "capital-loss": 0,
                "hours-per-week": 60,
                "native-country": "United-States"
            }
        }
    )
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.post("/inference/")
async def model_inference(item: CensusData):
    _CONFIG_PATH = Path(__file__).resolve().parent / "starter" / "config.yaml"

    with open(_CONFIG_PATH) as f:
        _config = yaml.safe_load(f)

    # Pfade in config.yaml sind relativ zu starter/starter/
    _BASE = _CONFIG_PATH.parent
    MODEL_PATH = (_BASE / _config["model_output_path"]).resolve()

    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    encoder = artifacts["encoder"]
    lb = artifacts["lb"]
    features = artifacts["features"]

    X = pd.DataFrame([item.model_dump(by_alias=True)])
    X_prep, _, _, _ = process_data(
        X,
        categorical_features=features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_prep)

    return {"prediction": lb.inverse_transform(preds)[0]}


"""

## API Creation
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

"""
