"""Chinese label generation with multilingual T5."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from device import get_default_device
from labels import ID_TO_LABEL, UNKNOWN_LABEL, parse_label

MAX_LABEL_TOKENS = 16


def make_prompt(text):
    return f"classify sarcasm: {text}"


class T5SarcasmClassifier:
    def __init__(self, model_name="google/mt5-base", device=None):
        self.device = torch.device(device) if device is not None else get_default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def predict_many(self, texts, max_length=64, num_beams=4):
        if not texts:
            return []
        # Preserve the original CPU workaround for generation on Apple MPS.
        generation_device = torch.device("cpu") if self.device.type == "mps" else self.device
        encoding = self.tokenizer(
            [make_prompt(text) for text in texts], return_tensors="pt",
            padding=True, truncation=True, max_length=max_length
        )
        encoding = {key: value.to(generation_device) for key, value in encoding.items()}
        self.model.eval()
        self.model.to(generation_device)
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoding, max_new_tokens=MAX_LABEL_TOKENS,
                    num_beams=num_beams, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            answers = self.tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True)
        finally:
            self.model.to(self.device)
        predictions = []
        for answer in answers:
            index = parse_label(answer)
            predictions.append(ID_TO_LABEL[index] if index is not None else UNKNOWN_LABEL)
        return predictions

    def predict(self, sentence, max_length=64, num_beams=4):
        return self.predict_many([sentence], max_length, num_beams)[0]

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
