import torch
from transformers import BertTokenizer, BertForSequenceClassification

class BertSarcasmClassifier:
    def __init__(self, model_name="bert-base-chinese", num_labels=2, device=None):
        """
        初始化：加载 tokenizer 与预训练模型，并将其移动到指定 device 上。
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if device is not None:
            self.model.to(device)
        self.device = device

    def predict(self, sentence: str, max_length: int = 32):
        """
        对单条句子做推理，返回标签（“反讽”/“非反讽”）以及置信度。
        """
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (1, 2)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()

        label = "反讽" if pred == 1 else "非反讽"
        return label, confidence

    def save(self, path: str):
        """
        保存模型权重到指定路径（只保存 state_dict）。
        """
        torch.save(self.model.state_dict(), path)

    def load(self, checkpoint_path: str):
        """
        从 checkpoint 加载模型权重。
        """
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)