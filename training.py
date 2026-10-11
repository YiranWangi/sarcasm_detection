"""Shared training loop with validation checkpoint selection and final test scoring."""

import argparse
import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np

from audit_data import audit_dataset
from data_utils import DEFAULT_DATA, split_data
from evaluation import evaluate_classifier
from labels import ID_TO_LABEL

MODEL_NAMES = {"bert": "bert-base-chinese", "t5": "google/mt5-base"}


def make_dataset(frame, tokenizer, model_type, max_length):
    import torch
    from torch.utils.data import Dataset

    class TextDataset(Dataset):
        def __init__(self):
            self.records = frame.to_dict("records")

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            text = record["text"]
            if model_type == "t5":
                from t5_classifier import make_prompt
                text = make_prompt(text)
            encoded = tokenizer(
                text, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt"
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            if model_type == "bert":
                item["labels"] = torch.tensor(record["label"], dtype=torch.long)
            else:
                from t5_classifier import MAX_LABEL_TOKENS
                target = tokenizer(
                    text_target=ID_TO_LABEL[record["label"]], truncation=True,
                    padding="max_length", max_length=MAX_LABEL_TOKENS,
                    return_tensors="pt"
                )
                labels = target["input_ids"].squeeze(0).clone()
                labels[labels == tokenizer.pad_token_id] = -100
                item["labels"] = labels
            return item

    return TextDataset()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_training(model_type):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=Path("runs") / model_type)
    parser.add_argument("--epochs", type=int, default=3 if model_type == "bert" else 10)
    parser.add_argument("--batch_size", type=int, default=16 if model_type == "bert" else 2)
    parser.add_argument("--learning_rate", type=float, default=2e-5 if model_type == "bert" else 3e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_length, args.num_beams) < 1:
        parser.error("epochs, batch_size, max_length and num_beams must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning_rate must be finite and positive.")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error("Use a new or empty output_dir to avoid mixing runs.")
    try:
        splits, split_audit = split_data(args.data_file, args.seed)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    try:
        import torch
        from torch.utils.data import DataLoader
        from device import get_default_device
        if model_type == "bert":
            from bert_classifier import BertSarcasmClassifier
            classifier_type = BertSarcasmClassifier
        else:
            from t5_classifier import T5SarcasmClassifier
            classifier_type = T5SarcasmClassifier
    except ModuleNotFoundError as error:
        parser.error(f"Missing dependency: {error.name}. Install requirements.txt first.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device or get_default_device()
    classifier = classifier_type(model_name=MODEL_NAMES[model_type], device=device)
    train_dataset = make_dataset(
        splits["train"], classifier.tokenizer, model_type, args.max_length
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed)
    )
    optimizer = torch.optim.AdamW(classifier.model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "model"
    config = {
        "model_type": model_type, "model_name": MODEL_NAMES[model_type],
        "data_file": str(args.data_file.resolve()), "device": str(device),
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "max_length": args.max_length,
        "num_beams": args.num_beams, "checkpoint_selection": "validation macro-F1, then accuracy",
        "data_sha256": split_audit["data_sha256"],
        "split_text_ids": split_audit["split_text_ids"],
        "versions": {name: version(name) for name in
                     ["torch", "transformers", "pandas", "scikit-learn", "numpy"]},
    }
    write_json(args.output_dir / "training_config.json", config)
    write_json(args.output_dir / "data_audit.json", audit_dataset(args.data_file, args.seed))
    print(f"Distinct texts: {split_audit['unique_texts']}; split sizes: {split_audit['split_sizes']}")
    if split_audit["unique_texts"] < 100:
        print("Small demonstration dataset: metrics do not establish real-world performance.")
    best_score = (-1.0, -1.0)
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        classifier.model.train()
        total_loss = 0.0
        examples = 0
        for batch in loader:
            batch = {key: value.to(classifier.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = classifier.model(**batch).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss; no test result will be reported.")
            loss.backward()
            optimizer.step()
            count = batch["input_ids"].shape[0]
            total_loss += float(loss.detach().cpu()) * count
            examples += count
        validation, _ = evaluate_classifier(
            classifier, splits["validation"], args.batch_size,
            args.max_length, args.num_beams
        )
        score = (validation["macro_f1"], validation["accuracy"])
        if score > best_score:
            classifier.save(checkpoint)
            best_score, best_epoch = score, epoch
        history.append({"epoch": epoch, "train_loss": total_loss / examples,
                        "validation": validation, "best_epoch": best_epoch})
        write_json(args.output_dir / "history.json", history)
        print(f"Epoch {epoch}: train loss={total_loss / examples:.4f}; "
              f"validation macro-F1={validation['macro_f1']:.4f}")

    classifier.load(checkpoint)
    test_metrics, test_predictions = evaluate_classifier(
        classifier, splits["test"], args.batch_size, args.max_length, args.num_beams
    )
    test_metrics["checkpoint_epoch"] = best_epoch
    test_metrics["data_sha256"] = split_audit["data_sha256"]
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    print("Held-out test:", json.dumps(test_metrics, ensure_ascii=False))
    print("Saved run:", args.output_dir)
