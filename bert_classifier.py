"""Chinese BERT sentence classification."""

from pathlib import Path

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from device import get_default_device
from labels import ID_TO_LABEL, LABEL_TO_ID


class BertSarcasmClassifier:
    def __init__(self, model_name="bert-base-chinese", num_labels=2, device=None):
        if num_labels != 2:
            raise ValueError("This task has two labels.")
        self.device = torch.device(device) if device is not None else get_default_device()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=2, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
        ).to(self.device)

    def _probabilities(self, texts, max_length):
        encoding = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        )
        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        self.model.eval()
        with torch.inference_mode():
            return torch.softmax(self.model(**encoding).logits, dim=-1).cpu()

    def predict_many(self, texts, max_length=64, num_beams=4):
        if not texts:
            return []
        probabilities = self._probabilities(texts, max_length)
        return [ID_TO_LABEL[index] for index in probabilities.argmax(dim=-1).tolist()]

    def predict(self, sentence, max_length=64):
        probabilities = self._probabilities([sentence], max_length)[0]
        index = int(probabilities.argmax())
        return ID_TO_LABEL[index], float(probabilities[index])

    def save(self, path):
        path = Path(path)
        if path.suffix == ".pth":
            torch.save(self.model.state_dict(), path)
        else:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def load(self, checkpoint_path):
        path = Path(checkpoint_path)
        if path.is_dir():
            self.tokenizer = BertTokenizer.from_pretrained(path)
            self.model = BertForSequenceClassification.from_pretrained(path).to(self.device)
        else:
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.to(self.device)
