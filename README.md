# Chinese sarcasm classification

A learning project comparing Chinese BERT sentence classification with multilingual T5 label generation.

- **BERT:** `bert-base-chinese`, with a two-class classification head.
- **MT5:** `google/mt5-base`, trained to generate `非反讽` or `反讽`.
- **Labels:** `0 = 非反讽` (non-sarcasm), `1 = 反讽` (sarcasm).

## Dataset and evaluation limits

The included `scrcasm_data.xlsx` has 4,000 rows but only **20 distinct texts**, with 10 texts per class. The original row-level split put copies of every validation and test text in the training set. The previously reported 100% validation accuracy therefore does not establish generalization.

The revised preprocessing normalizes Unicode and whitespace, rejects conflicting labels, removes duplicate texts, and then makes a stratified split with seed 42:

| Split | Distinct texts |
| --- | ---: |
| Training | 16 |
| Validation | 2 |
| Test | 2 |

See [data_audit.json](data_audit.json) for the file hash, split membership and original overlap counts. The workbook is preserved unchanged. A two-example test set is useful for checking the pipeline, but too small for a meaningful model comparison. Deduplication also does not rule out similarities between templates or sources.

**This repository is a demonstration, not a validated sarcasm detector.** No new neural training results are reported here. A credible benchmark would need a larger dataset with documented provenance and annotation, source-aware splitting where applicable, and repeated runs.

## Run

Use Python 3.10 or 3.11 in a virtual environment. The first training run downloads pretrained weights from Hugging Face; MT5-base may need substantial memory.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python audit_data.py --output data_audit.json
python -m unittest discover -s tests -v
python train_bert.py --output_dir runs/bert
python train_t5.py --output_dir runs/t5
```

Both training scripts accept `--data_file`, `--seed`, `--epochs`, `--batch_size`, `--learning_rate`, `--max_length` and `--device cpu|cuda|mps`. MT5 generation also uses `--num_beams`. For another dataset, supply an Excel or CSV file with `text` and `label` columns and at least 10 distinct texts per class. Use a new output directory for each run.

A run saves:

| File | Purpose |
| --- | --- |
| `model/` | Model and tokenizer from the best validation epoch |
| `training_config.json` | Parameters, dependency versions, data hash and split IDs |
| `data_audit.json` | Duplicate counts and split audit |
| `history.json` | Training loss and validation metrics by epoch |
| `test_metrics.json` | Final held-out test accuracy, macro-F1, invalid rate and confusion matrix |
| `test_predictions.csv` | Per-example test predictions |

Checkpoint selection uses validation macro-F1, breaking ties with accuracy. The test split is evaluated once after loading that checkpoint. Invalid MT5 answers are counted as errors and displayed as `未知`; they are never silently assigned to a class. MT5 target padding is excluded from the training loss.

## Predict

```bash
python predict.py --model_type bert --checkpoint_path runs/bert/model --text "今天天气真好。"
python predict.py --model_type t5 --checkpoint_path runs/t5/model --text "今天天气真好。"
```

Prediction uses the saved training settings when available. BERT's displayed probability is a model output, not a calibrated confidence estimate. Legacy BERT `.pth` weights remain usable for inference, but should be retrained with the corrected split before any evaluation claim.

## Code

`data_utils.py` handles validation and splitting; `training.py` runs both training paths; `evaluation.py` uses the same prediction methods as inference. Tests cover duplicate leakage, label consistency, invalid predictions and batch alignment. Optional offline model-wrapper tests require PyTorch and Transformers; they do not train or download a model.

`advanced_models.py` contains separate BERT-CNN and BERT-CRF experiments. They are not part of the comparison above; the CRF path also requires an optional `torchcrf` dependency.
