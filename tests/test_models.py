"""Offline wrapper tests; require torch and transformers, but no downloaded models."""

import unittest
from unittest.mock import patch

try:
    import torch
    import transformers
    HAS_MODELS = True
except ModuleNotFoundError:
    HAS_MODELS = False


@unittest.skipUnless(HAS_MODELS, "torch and transformers are not installed")
class ModelTests(unittest.TestCase):
    def classifier(self, answers, fail=False):
        from t5_classifier import T5SarcasmClassifier

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, texts, **kwargs):
                count = len(texts)
                return {"input_ids": torch.ones((count, 3), dtype=torch.long),
                        "attention_mask": torch.ones((count, 3), dtype=torch.long)}

            def batch_decode(self, generated, **kwargs):
                return answers

        class Model:
            def __init__(self):
                self.moves = []

            def eval(self):
                return self

            def to(self, device):
                self.moves.append(str(device))
                return self

            def generate(self, **kwargs):
                if fail:
                    raise RuntimeError("generation failed")
                self.generation_args = kwargs
                return torch.ones((len(answers), 3), dtype=torch.long)

        classifier = T5SarcasmClassifier.__new__(T5SarcasmClassifier)
        classifier.device = torch.device("cpu")
        classifier.tokenizer = Tokenizer()
        classifier.model = Model()
        return classifier

    def test_mt5_decodes_chinese_labels_and_preserves_unknown(self):
        classifier = self.classifier(["反讽", "非反讽", "", "<extra_id_0>"])
        self.assertEqual(
            classifier.predict_many(["a", "b", "c", "d"]),
            ["反讽", "非反讽", "未知", "未知"]
        )
        self.assertGreater(classifier.model.generation_args["max_new_tokens"], 2)
        self.assertEqual(classifier.model.moves[-1], "cpu")

    def test_generation_failure_restores_device(self):
        classifier = self.classifier([], fail=True)
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            classifier.predict_many(["a"])
        self.assertEqual(classifier.model.moves, ["cpu", "cpu"])

    def test_mt5_load_uses_auto_model_and_slow_tokenizer(self):
        classifier = self.classifier(["反讽"])
        with patch("t5_classifier.AutoTokenizer.from_pretrained",
                   return_value=classifier.tokenizer) as tokenizer, patch(
                       "t5_classifier.AutoModelForSeq2SeqLM.from_pretrained",
                       return_value=classifier.model) as model:
            classifier.load("local-checkpoint")
        tokenizer.assert_called_once_with("local-checkpoint", use_fast=False)
        model.assert_called_once_with("local-checkpoint")

    def test_mt5_padding_is_ignored_in_target_loss(self):
        import pandas as pd
        from training import make_dataset

        class Tokenizer:
            pad_token_id = 0

            def __call__(self, text=None, text_target=None, **kwargs):
                ids = [3, 1, 0, 0] if text_target is not None else [2, 1, 0, 0]
                return {"input_ids": torch.tensor([ids]),
                        "attention_mask": torch.tensor([[1, 1, 0, 0]])}
        frame = pd.DataFrame([{"text": "example", "label": 1}])
        dataset = make_dataset(frame, Tokenizer(), "t5", 4)
        self.assertEqual(dataset[0]["labels"].tolist(), [3, 1, -100, -100])


if __name__ == "__main__":
    unittest.main()
