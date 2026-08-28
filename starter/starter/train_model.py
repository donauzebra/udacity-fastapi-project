"""Train and evaluate the census salary model, then stores the trained model."""

from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
try:
    from .ml.data import process_data
    from .ml.model import train_model, inference, compute_model_metrics, save_model, compute_model_slice_metrics
except ImportError:
    # Fallback for direct script execution from the package directory.
    from ml.data import process_data
    from ml.model import train_model, inference, compute_model_metrics, save_model, compute_model_slice_metrics

# Configuration
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def _load_config_and_paths():
    """Load config and resolve input/output paths plus validation mode."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = (BASE_DIR / config["data_input_path"]).resolve()
    model_path = (BASE_DIR / config["model_output_path"]).resolve()
    return data_path, model_path, config["validation_method"]


def _load_and_clean_data(data_path):
    """Load input CSV data and apply basic column cleanup."""
    data = pd.read_csv(data_path, skipinitialspace=True)
    data.columns = data.columns.str.strip()
    return data


def _split_data(data, validation_method):
    """Split data based on the configured validation strategy."""
    if validation_method == "train_test":
        return train_test_split(data, test_size=0.20)
    if validation_method == "kfold":
        raise ValueError("K-Fold validation method is not implemented yet")
    raise ValueError(f"Unknown validation method: {validation_method}")


def run_training():
    """Run the full training pipeline and persist the trained model.

    Returns
    -------
    tuple[float, float, float]
        Precision, recall, and F1 score on the holdout split.

    Raises
    ------
    ValueError
        If the configured validation method is unsupported.
    """
    data_path, model_path, validation_method = _load_config_and_paths()
    data = _load_and_clean_data(data_path)
    train, test = _split_data(data, validation_method)
    x_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    x_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    model = train_model(x_train, y_train)
    preds = inference(model, x_test)
    metrics = compute_model_metrics(y_test, preds)

    save_model(model, encoder, lb, CAT_FEATURES, model_path)

    slice_output_path = (BASE_DIR / "slice_output.txt").resolve()
    for feature in CAT_FEATURES:
        compute_model_slice_metrics(
            model, encoder, lb, test, CAT_FEATURES, feature, output_path=slice_output_path
        )

    return metrics


if __name__ == "__main__":
    precision, recall, f1 = run_training()
    print("Training finished!")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
