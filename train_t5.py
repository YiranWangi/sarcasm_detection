# train_t5.py

import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import AdamW
from torch.optim import AdamW
from device import get_default_device
from t5_classifier import T5SarcasmClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


# -------------- 1. 自定义 Dataset ----------
class T5SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_src_len=64, max_tgt_len=4):
        """
        texts: List[str]，labels: List[str]（"反讽" 或 "非反讽"）
        tokenizer: T5Tokenizer / MT5Tokenizer
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        src = f"classify sarcasm: {self.texts[idx]}"
        tgt = self.labels[idx]
        src_encoding = self.tokenizer(
            src,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_src_len
        )
        with self.tokenizer.as_target_tokenizer():
            tgt_encoding = self.tokenizer(
                tgt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_tgt_len
            )
        item = {
            "input_ids":      src_encoding["input_ids"].squeeze(0),
            "attention_mask": src_encoding["attention_mask"].squeeze(0),
            "labels":         tgt_encoding["input_ids"].squeeze(0)
        }
        return item

# -------------- 2. 准备数据（从 sarcasm_data.xlsx 加载并切分） ----------
df = pd.read_excel("scrcasm_data.xlsx")
# 随机打乱
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 把 label 0/1 映射为“非反讽”/“反讽”
df["label"] = df["label"].map({0: "非反讽", 1: "反讽"})

train_df, temp_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

train_texts  = train_df["text"].tolist()
train_labels = train_df["label"].tolist()

val_texts  = val_df["text"].tolist()
val_labels = val_df["label"].tolist()

test_texts  = test_df["text"].tolist()
test_labels = test_df["label"].tolist()

print(f"Train/Val/Test = {len(train_texts)}/{len(val_texts)}/{len(test_texts)}")

# -------------- 3. 初始化模型、优化器、DataLoader ----------
device = get_default_device()
print("Using device:", device)

model = T5SarcasmClassifier(model_name="google/mt5-base", device=device)
tokenizer = model.tokenizer

train_dataset = T5SarcasmDataset(train_texts, train_labels, tokenizer, max_src_len=64, max_tgt_len=4)
val_dataset   = T5SarcasmDataset(val_texts,   val_labels,   tokenizer, max_src_len=64, max_tgt_len=4)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=4)

optimizer = AdamW(model.model.parameters(), lr=3e-5)

# -------------- 4. Training Loop ----------
num_epochs = 10
for epoch in range(num_epochs):
    model.model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

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

    # -------------- 验证阶段（CPU 上执行 generate） ----------
    model.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids_gpu      = batch["input_ids"].to(device)
            attention_mask_gpu = batch["attention_mask"].to(device)
            labels_gpu         = batch["labels"].to(device)

            # 临时将模型和张量搬到 CPU，执行 generate()
            model.model.to("cpu")
            input_ids_cpu      = input_ids_gpu.to("cpu")
            attention_mask_cpu = attention_mask_gpu.to("cpu")

            generated_ids_cpu = model.model.generate(
                input_ids=input_ids_cpu,
                attention_mask=attention_mask_cpu,
                max_length=4,
                num_beams=4,
                early_stopping=True,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )

            # 将模型搬回 MPS（或指定 device）
            model.model.to(device)

            # 解码预测结果
            preds = [
                model.tokenizer.decode(g, skip_special_tokens=True)
                for g in generated_ids_cpu
            ]
            # 解码标签
            targets = [
                model.tokenizer.decode(t.cpu(), skip_special_tokens=True)
                for t in labels_gpu
            ]

            for p, t in zip(preds, targets):
                if p == t:
                    correct += 1
                total += 1

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

# -------------- 5. 保存模型 ----------
model.save("best_t5_model")
print("Saved best_t5_model/")

