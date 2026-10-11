"""Predict with a saved checkpoint; these examples are not an evaluation."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", choices=["bert", "t5"], required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--text", nargs="+", help="One or more quoted sentences.")
    parser.add_argument("--max_length", type=int, help="Defaults to the training setting, or 64.")
    parser.add_argument("--num_beams", type=int, help="MT5 only; defaults to the training setting, or 4.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    if not args.checkpoint_path.exists():
        parser.error("The checkpoint path does not exist.")
    config_path = args.checkpoint_path.parent / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if config.get("model_type", args.model_type) != args.model_type:
        parser.error("The checkpoint was trained with a different model type.")
    max_length = args.max_length if args.max_length is not None else config.get("max_length", 64)
    num_beams = args.num_beams if args.num_beams is not None else config.get("num_beams", 4)
    if max_length < 1 or num_beams < 1:
        parser.error("max_length and num_beams must be positive.")

    from data_utils import normalize_text
    from device import get_default_device

    device = args.device or get_default_device()
    print("Using device:", device)
    if args.model_type == "bert":
        from bert_classifier import BertSarcasmClassifier
        if args.checkpoint_path.is_dir():
            classifier = BertSarcasmClassifier(model_name=str(args.checkpoint_path), device=device)
        else:
            classifier = BertSarcasmClassifier(device=device)
            classifier.load(args.checkpoint_path)
    else:
        from t5_classifier import T5SarcasmClassifier
        if not args.checkpoint_path.is_dir():
            parser.error("MT5 requires a saved model directory.")
        classifier = T5SarcasmClassifier(model_name=str(args.checkpoint_path), device=device)

    samples = args.text or [
        "你好棒棒哦，椎间盘都没有你突出。",
        "今天天气真好。",
        "你这人真聪明，一点都不笨。",
    ]
    for sentence in samples:
        text = normalize_text(sentence)
        if args.model_type == "bert":
            label, confidence = classifier.predict(text, max_length=max_length)
            print(f"{sentence}\n  {label} (model probability: {confidence:.3f})")
        else:
            label = classifier.predict(text, max_length=max_length, num_beams=num_beams)
            print(f"{sentence}\n  {label}")


if __name__ == "__main__":
    main()
