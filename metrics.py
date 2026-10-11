"""Score invalid generated labels as errors, rather than as a default class."""

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(expected, predicted):
    expected, predicted = list(expected), list(predicted)
    if not expected or len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must have the same nonzero length.")
    if any(label not in (0, 1) for label in expected):
        raise ValueError("Reference labels must be 0 or 1.")
    if any(label is not None and label not in (0, 1) for label in predicted):
        raise ValueError("Predictions must be 0, 1 or None.")
    scored = [-1 if label is None else label for label in predicted]
    matrix = [[0, 0, 0], [0, 0, 0]]
    for actual, guess in zip(expected, scored):
        matrix[actual][2 if guess == -1 else guess] += 1
    invalid = scored.count(-1)
    return {
        "n": len(expected), "accuracy": float(accuracy_score(expected, scored)),
        "macro_f1": float(f1_score(expected, scored, labels=[0, 1],
                                   average="macro", zero_division=0)),
        "invalid_predictions": invalid, "invalid_prediction_rate": invalid / len(expected),
        "confusion_matrix": matrix,
        "confusion_rows": ["非反讽", "反讽"],
        "confusion_columns": ["非反讽", "反讽", "未知"],
    }
