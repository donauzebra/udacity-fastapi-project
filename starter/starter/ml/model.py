import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from ml.data import process_data


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
    model : RandomForestClassifier
        Trained machine learning model.
    """

    rfc = RandomForestClassifier(random_state=42)

    param_dist = {
        'n_estimators': [200, 300],
        'max_features': ['sqrt'],
        'max_depth': [5, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'criterion': ['gini']
    }

    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_dist,
        n_iter=12,
        cv=5,
        scoring="f1",
        random_state=42,
        error_score='raise'
    )

    cv_rfc.fit(X_train, y_train)

    return cv_rfc.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

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


def save_model(model, encoder, lb, features, path):
    """Saves model, encoders, features and label binarizer to disk.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted categorical encoder.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted label binarizer.
    features : list[str]
        List of categorical feature names used during training.
    path : str or Path
        Destination file path (pickle format).
    """
    with open(path, "wb") as f:
        pickle.dump({"model": model, "encoder": encoder, "lb": lb, "features": features}, f)


def compute_model_slice_metrics(model, encoder, lb, df, features, sel_feature, output_path="slice_output.txt"):

    if sel_feature not in features:
        raise ValueError(f"'{sel_feature}' is not in features: {features}")
    
    feature_values = df[sel_feature].unique()

    content = []

    for feature_value in feature_values:
        test = df[df[sel_feature] == feature_value]


        x_test, y_test, _, _ = process_data(
            test,
            categorical_features=features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, x_test)
        precision, recall, f1_score = compute_model_metrics(y_test, preds)

        content.append(
            {
                "feature": sel_feature,
                "feature_value": feature_value,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        )

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"feature: {sel_feature}\n")
            f.write(f"feature_value: {feature_value}\n")
            f.write(f"precision: {precision}\n")
            f.write(f"recall: {recall}\n")
            f.write(f"f1_score: {f1_score}\n")
            f.write("\n")
    
    return content

