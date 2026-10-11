"""Use the same prediction method for validation, held-out testing and inference."""

import pandas as pd

from labels import parse_label
from metrics import classification_metrics


def evaluate_classifier(classifier, frame, batch_size=16, max_length=64, num_beams=4):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    rows = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start:start + batch_size]
        answers = classifier.predict_many(
            batch.text.tolist(), max_length=max_length, num_beams=num_beams
        )
        if len(answers) != len(batch):
            raise ValueError("The classifier returned the wrong number of predictions.")
        for (_, record), answer in zip(batch.iterrows(), answers):
            rows.append({"text_id": record.text_id, "text": record.text,
                         "label": int(record.label), "prediction": parse_label(answer),
                         "generated_label": answer})
    predictions = pd.DataFrame(rows)
    metrics = classification_metrics(
        [row["label"] for row in rows], [row["prediction"] for row in rows]
    )
    return metrics, predictions
