"""Validate and split distinct texts before any model is trained."""

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA = Path(__file__).with_name("scrcasm_data.xlsx")


def normalize_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def read_data(path):
    path = Path(path)
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Data must contain text and label columns.")
    frame = frame[["text", "label"]].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("Data is empty or contains missing text/labels.")
    if not frame.text.map(lambda value: isinstance(value, str)).all():
        raise ValueError("Every text must be a string.")
    frame["text"] = frame.text.map(normalize_text)
    if frame.text.eq("").any():
        raise ValueError("Empty text after normalization.")
    labels = pd.to_numeric(frame.label, errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Labels must be 0 (non-sarcasm) or 1 (sarcasm).")
    frame["label"] = labels.astype(int)
    conflicts = frame.groupby("text").label.nunique().gt(1)
    if conflicts.any():
        raise ValueError(f"{int(conflicts.sum())} texts have conflicting labels.")
    frame["text_id"] = frame.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    return frame


def split_data(path=DEFAULT_DATA, seed=42):
    frame = read_data(path)
    unique = frame.drop_duplicates("text").reset_index(drop=True)
    class_counts = unique.label.value_counts()
    if set(class_counts.index) != {0, 1} or class_counts.min() < 10:
        raise ValueError("An 80/10/10 stratified split needs at least 10 distinct texts per class.")
    train, remainder = train_test_split(
        unique, test_size=0.2, random_state=seed, stratify=unique.label
    )
    validation, test = train_test_split(
        remainder, test_size=0.5, random_state=seed, stratify=remainder.label
    )
    splits = {"train": train.reset_index(drop=True),
              "validation": validation.reset_index(drop=True),
              "test": test.reset_index(drop=True)}
    sets = [set(part.text_id) for part in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Text overlap between splits.")
    audit = {
        "data_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "seed": seed, "original_rows": len(frame), "unique_texts": len(unique),
        "duplicate_rows_removed": len(frame) - len(unique),
        "unique_label_counts": {str(k): int(v) for k, v in class_counts.items()},
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "split_text_ids": {name: sorted(part.text_id.tolist()) for name, part in splits.items()},
        "normalization": "NFKC Unicode normalization, collapsed whitespace, outer whitespace stripped",
        "pairwise_text_overlap": 0,
    }
    return splits, audit
