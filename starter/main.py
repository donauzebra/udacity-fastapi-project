# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

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
