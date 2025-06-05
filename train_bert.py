# train_bert.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from transformers import AdamW
from torch.optim import AdamW
from device import get_default_device
from bert_classifier import BertSarcasmClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------- 1. 自定义 Dataset ----------
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
        """
        texts: List[str]，labels: List[int]（0/1）
        tokenizer: BertTokenizer
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sentence = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        return item

# -------------- 2. 准备数据（从 sarcasm_data.xlsx 加载并切分） ----------
df = pd.read_excel("scrcasm_data.xlsx",engine = "openpyxl")
# 随机打乱
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 先把 data 分成 train (80%) 和 temp (20%)
train_df, temp_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label"]
)
# 再把 temp 分成 valid (50% of temp => 10% 总量) 和 test (剩下 10%)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

# 生成最终的列表
train_texts  = train_df["text"].tolist()
train_labels = train_df["label"].astype(int).tolist()  # 0 或 1

val_texts  = val_df["text"].tolist()
val_labels = val_df["label"].astype(int).tolist()

test_texts  = test_df["text"].tolist()
test_labels = test_df["label"].astype(int).tolist()

print(f"训练集样本数：{len(train_texts)}，验证集样本数：{len(val_texts)}，测试集样本数：{len(test_texts)}")

# -------------- 3. 初始化模型、优化器、DataLoader ----------
device = get_default_device()
print("Using device:", device)

model = BertSarcasmClassifier(
    model_name="bert-base-chinese",
    num_labels=2,
    device=device
)
tokenizer = model.tokenizer

train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer, max_length=32)
val_dataset   = SarcasmDataset(val_texts,   val_labels,   tokenizer, max_length=32)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32)

optimizer = AdamW(model.model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# -------------- 4. Training Loop ----------
num_epochs = 3
for epoch in range(num_epochs):
    model.model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}  Train Loss: {avg_train_loss:.4f}")

    # -------------- 验证阶段 ----------
    model.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

# -------------- 5. 保存模型 ----------
model.save("best_bert.pth")
print("Saved best_bert.pth")

