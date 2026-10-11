"""Report duplicates, legacy split overlap and the new distinct-text split."""

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from data_utils import DEFAULT_DATA, read_data, split_data


def audit_dataset(path=DEFAULT_DATA, seed=42):
    splits, audit = split_data(path, seed)
    legacy = read_data(path).sample(frac=1, random_state=seed).reset_index(drop=True)
    train, remainder = train_test_split(
        legacy, test_size=0.2, random_state=seed, stratify=legacy.label
    )
    validation, test = train_test_split(
        remainder, test_size=0.5, random_state=seed, stratify=remainder.label
    )
    audit["legacy_split_overlap"] = {
        "validation_rows_seen_in_training": int(validation.text_id.isin(train.text_id).sum()),
        "validation_rows": len(validation),
        "test_rows_seen_in_training": int(test.text_id.isin(train.text_id).sum()),
        "test_rows": len(test),
    }
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_dataset(args.data_file, args.seed)
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
