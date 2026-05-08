# Script to train machine learning model.

from pathlib import Path
import pandas as pd
import pickle
import yaml
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Configuration
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

data_path = (BASE_DIR / config["data_input_path"]).resolve()
model_path = (BASE_DIR / config["model_output_path"]).resolve()

# Add code to load in the data.
data = pd.read_csv(data_path)
data.columns = (data.columns.str.strip())

# Optional enhancement, use K-fold cross validation instead of a train-test split.
if config["validation_method"] == "train_test":
    train, test = train_test_split(data, test_size=0.20)
elif config["validation_method"] == "kfold":
    raise ValueError("K-Fold validation method is not implemented yet")
else:
    raise ValueError(f"Unkown validation method: {config["validation_method"]}")

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

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

with open(model_path, "wb") as file:
    pickle.dump(model, file)
