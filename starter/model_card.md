# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained with scikit-learn. Hyperparameters were selected via RandomizedSearchCV with 5-fold cross-validation optimizing for F1-score.

## Intended Use
This model is intended for educational purposes to demonstrate a machine learning pipeline using Census data.

## Training Data
The model was trained on a Census Income dataset. The full dataset contains ~32,500 records; 80% were used for training. It contains 14 features including age, education, occupation, and marital status. Categorical features were one-hot encoded.

## Evaluation Data
The model was evaluated on a randomly held-out 20% split of the Census Income dataset, using the same preprocessing pipeline applied during training.

## Metrics
The model is evaluated on a test set (20% split) achieving the following metrics:
- Precision: 0.7951
- Recall:    0.5522
- F1-Score:  0.6518

## Ethical Considerations
The dataset contains sensitive demographic attributes such as race, sex, and native country. These features may introduce or amplify bias in predictions. The model should not be used in any context where its output could disadvantage individuals based on these characteristics.

## Caveats and Recommendations
This is a learning project intended to practice MLOps concepts such as data processing, model training, and API deployment. No claims are made about the model's accuracy or suitability for any real-world task.
