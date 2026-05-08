"""Smoke tests for the training and inference pipeline."""

import pandas as pd
import pytest


@pytest.fixture(name="sample_data")
def fixture_sample_data():
    """Provide a small representative census-like dataset for tests."""
    return pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28, 44, 36, 41, 32, 47],
            "hours-per-week": [40, 13, 40, 40, 55, 45, 38, 60, 35, 50],
            "workclass": [
                "State-gov",
                "Self-emp-not-inc",
                "Private",
                "Private",
                "Private",
                "Federal-gov",
                "Private",
                "State-gov",
                "Local-gov",
                "Self-emp-inc",
            ],
            "education": [
                "Bachelors",
                "Bachelors",
                "HS-grad",
                "11th",
                "Masters",
                "Assoc-voc",
                "Some-college",
                "Doctorate",
                "HS-grad",
                "Bachelors",
            ],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Never-married",
                "Separated",
                "Married-civ-spouse",
                "Never-married",
                "Married-civ-spouse",
            ],
            "occupation": [
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
                "Handlers-cleaners",
                "Prof-specialty",
                "Tech-support",
                "Sales",
                "Exec-managerial",
                "Craft-repair",
                "Prof-specialty",
            ],
            "relationship": [
                "Not-in-family",
                "Husband",
                "Not-in-family",
                "Husband",
                "Wife",
                "Not-in-family",
                "Unmarried",
                "Husband",
                "Own-child",
                "Husband",
            ],
            "race": [
                "White",
                "White",
                "White",
                "Black",
                "White",
                "Black",
                "White",
                "Asian-Pac-Islander",
                "White",
                "White",
            ],
            "sex": [
                "Male",
                "Male",
                "Male",
                "Male",
                "Female",
                "Female",
                "Male",
                "Male",
                "Female",
                "Male",
            ],
            "native-country": [
                "United-States",
                "United-States",
                "United-States",
                "United-States",
                "India",
                "United-States",
                "Mexico",
                "United-States",
                "United-States",
                "Canada",
            ],
            "salary": [
                ">50K",
                "<=50K",
                "<=50K",
                "<=50K",
                ">50K",
                "<=50K",
                "<=50K",
                ">50K",
                ">50K",
                ">50K",
            ],
        }
    )


@pytest.fixture(name="sample_config")
def fixture_sample_config():
    """Provide categorical feature configuration used by process_data."""
    return {
        "categorical_features": [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
    }


def test_pytest_runs():
    """Sanity-check that pytest can discover and execute tests."""
    assert True

def test_training_pipeline_runs(sample_data, sample_config):
    """Validate end-to-end training and inference on fixture data."""
    from starter.ml.data import process_data
    from starter.ml.model import train_model, inference

    x_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=sample_config["categorical_features"],
        label="salary",
        training=True
    )

    x_test, y_test, _, _ = process_data(
        sample_data,
        categorical_features=sample_config["categorical_features"],
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    model = train_model(x_train, y_train)
    preds = inference(model, x_test)

    assert model is not None
    assert len(preds) == len(y_test)
    assert set(preds.tolist()).issubset({0, 1})
