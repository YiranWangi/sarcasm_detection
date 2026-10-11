"""Regression tests for data leakage, invalid labels and prediction alignment."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from audit_data import audit_dataset
from data_utils import DEFAULT_DATA, normalize_text, read_data, split_data
from evaluation import evaluate_classifier
from labels import parse_label
from metrics import classification_metrics


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "data.csv"
        self.rows = [{"text": f"sample {label} {index}", "label": label}
                     for label in (0, 1) for index in range(10)]

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows):
        pd.DataFrame(rows).to_csv(self.path, index=False)
        return self.path

    def test_normalization(self):
        self.assertEqual(normalize_text("  Ａ\t B\n"), "A B")

    def test_deduplication_precedes_split(self):
        splits, audit = split_data(self.write(self.rows * 3))
        self.assertEqual(audit["original_rows"], 60)
        self.assertEqual(audit["unique_texts"], 20)
        self.assertEqual(audit["split_sizes"], {"train": 16, "validation": 2, "test": 2})
        sets = [set(frame.text) for frame in splits.values()]
        self.assertEqual(len(set.union(*sets)), 20)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(sets[i] & sets[j])

    def test_split_is_repeatable(self):
        path = self.write(self.rows * 2)
        first, audit = split_data(path)
        second, again = split_data(path)
        self.assertEqual(audit, again)
        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_normalized_duplicates_with_conflicting_labels_fail(self):
        path = self.write([{"text": "Ａ  B", "label": 0}, {"text": "A B", "label": 1}])
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            read_data(path)

    def test_invalid_labels_fail(self):
        for value in (2, -1, 0.5, "not a label"):
            with self.subTest(value=value):
                path = self.write([{"text": "example", "label": value}])
                with self.assertRaisesRegex(ValueError, "Labels must be"):
                    read_data(path)

    def test_missing_or_empty_text_fails(self):
        for text in (None, "   "):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    read_data(self.write([{"text": text, "label": 0}]))

    def test_missing_columns_fail(self):
        with self.assertRaisesRegex(ValueError, "columns"):
            read_data(self.write([{"sentence": "example", "label": 0}]))

    def test_too_few_distinct_texts_fail(self):
        with self.assertRaisesRegex(ValueError, "distinct texts per class"):
            split_data(self.write(self.rows[:4] * 100))

    def test_bundled_data_audit_matches_saved_report(self):
        actual = audit_dataset(DEFAULT_DATA)
        saved = json.loads(DEFAULT_DATA.with_name("data_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(actual, saved)
        self.assertEqual(actual["unique_texts"], 20)
        self.assertEqual(actual["duplicate_rows_removed"], 3980)
        self.assertEqual(actual["legacy_split_overlap"]["test_rows_seen_in_training"], 400)


class LabelAndMetricTests(unittest.TestCase):
    def test_generated_labels_match_training_labels(self):
        self.assertEqual(parse_label(" 反讽 "), 1)
        self.assertEqual(parse_label("非反讽"), 0)

    def test_invalid_generations_do_not_default_to_non_sarcasm(self):
        for answer in ("", "<extra_id_0>", "unknown", "这可能是反讽", None):
            with self.subTest(answer=answer):
                self.assertIsNone(parse_label(answer))

    def test_invalid_prediction_is_an_error(self):
        metrics = classification_metrics([0, 1], [None, 1])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["invalid_predictions"], 1)
        self.assertEqual(metrics["confusion_matrix"], [[0, 0, 1], [0, 1, 0]])

    def test_empty_or_misaligned_evaluation_fails(self):
        for expected, predicted in (([], []), ([0], []), ([0], [2])):
            with self.subTest(expected=expected, predicted=predicted):
                with self.assertRaises(ValueError):
                    classification_metrics(expected, predicted)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"text_id": "a", "text": "first", "label": 0},
            {"text_id": "b", "text": "second", "label": 1},
            {"text_id": "c", "text": "third", "label": 1},
        ])

    def test_batches_preserve_alignment_and_score_unknown(self):
        class Classifier:
            def predict_many(self, texts, **kwargs):
                return [{"first": "非反讽", "second": "反讽", "third": "未知"}[text]
                        for text in texts]
        metrics, predictions = evaluate_classifier(Classifier(), self.frame, batch_size=2)
        self.assertEqual(predictions.text_id.tolist(), ["a", "b", "c"])
        self.assertEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["invalid_predictions"], 1)

    def test_wrong_prediction_count_fails(self):
        class Classifier:
            def predict_many(self, texts, **kwargs):
                return []
        with self.assertRaisesRegex(ValueError, "wrong number"):
            evaluate_classifier(Classifier(), self.frame)


if __name__ == "__main__":
    unittest.main()
