import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "is_spam"
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def validate_data(frame, feature_names, max_rows=None):
    """Validate labelled data and return inputs in the expected column order."""
    if frame.empty:
        raise ValueError("The dataset contains no rows.")

    if max_rows is not None and len(frame) > max_rows:
        raise ValueError(f"Use at most {max_rows:,} test rows.")

    expected = set(feature_names) | {TARGET}
    missing = expected - set(frame.columns)
    extra = set(frame.columns) - expected

    if missing or extra:
        raise ValueError(
            f"Column mismatch. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(extra)}"
        )

    try:
        features = frame.loc[:, feature_names].apply(
            pd.to_numeric, errors="raise"
        )
        target = pd.to_numeric(frame[TARGET], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Features and target must be numeric.") from error

    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("is_spam must contain only 0 or 1.")

    values = features.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Infinite input values are not supported.")

    return features, target.astype(int)


def calculate_metrics(actual, predicted, scores):
    """Use class predictions for five metrics and continuous scores for AUC."""
    auc = (
        roc_auc_score(actual, scores)
        if len(np.unique(actual)) == 2
        else float("nan")
    )

    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "AUC": float(auc),
        "Precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "Recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(actual, predicted)),
    }


def evaluate_model(model, features, target):
    """Evaluate an already-trained model without fitting it."""
    predictions = model.predict(features)

    if hasattr(model, "predict_proba"):
        positive_column = list(model.classes_).index(1)
        scores = model.predict_proba(features)[:, positive_column]
        score_kind = "Probability of spam"
    else:
        scores = model.decision_function(features)
        score_kind = "Spam decision score (not a probability)"

    return {
        "metrics": calculate_metrics(target, predictions, scores),
        "predictions": predictions,
        "scores": scores,
        "score_kind": score_kind,
        "confusion_matrix": confusion_matrix(
            target, predictions, labels=[0, 1]
        ),
        "classification_report": classification_report(
            target,
            predictions,
            labels=[0, 1],
            target_names=["Non-spam", "Spam"],
            output_dict=True,
            zero_division=0,
        ),
    }


if __name__ == "__main__":
    # Four artificial examples with independently known metric values.
    actual = [0, 0, 1, 1]
    predicted = [0, 1, 1, 1]
    scores = [0.1, 0.7, 0.8, 0.9]

    expected = {
        "Accuracy": 0.75,
        "AUC": 1.0,
        "Precision": 2 / 3,
        "Recall": 1.0,
        "F1": 0.8,
        "MCC": 1 / np.sqrt(3),
    }

    results = calculate_metrics(actual, predicted, scores)

    print("Metric check using four artificial examples:")
    for name in METRICS:
        assert np.isclose(results[name], expected[name]), name
        print(f"{name}: {results[name]:.4f}")

    print("\nMetric calculation check passed.")