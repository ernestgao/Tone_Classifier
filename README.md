# Politeness Tone Classifier (DeBERTa)

This project fine-tunes `microsoft/deberta-v3-base` for 3-way tone classification:
- `impolite` (API label `-1`)
- `neutral` (API label `0`)
- `polite` (API label `1`)

It supports:
- single-run training
- multi-run tuning to target higher accuracy
- saving both Hugging Face model directory and Torch `.pt`
- inference from raw text

## 1) Environment (on your NVIDIA machine)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 2) Data format

Use CSV/JSON/JSONL/Parquet with:
- text column (default: `text`)
- label column (default: `label`)

Accepted label values:
- integers: `-1`, `0`, `1`
- or strings: `impolite`, `neutral`, `polite`

Example CSV:

```csv
text,label
"Could you please share the logs when you get a chance?",1
"Send it now.",-1
"I need the project status update.",0
```

### Auto-download Stanford politeness data (recommended)

If you do not already have `train.csv/valid.csv/test.csv`, generate them automatically:

```bash
python -m tone_classifier.prepare_data --output_dir data
```

This downloads Stanford's Wikipedia politeness corpus via ConvoKit and creates:
- `data/train.csv`
- `data/valid.csv`
- `data/test.csv`

## 3) Single training run (balanced speed + quality)

```bash
python -m tone_classifier.train \
  --train_file data/train.csv \
  --validation_file data/valid.csv \
  --test_file data/test.csv \
  --text_column text \
  --label_column label \
  --model_name microsoft/deberta-v3-base \
  --max_length 128 \
  --num_train_epochs 8 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.06 \
  --eval_steps 100 \
  --early_stopping_patience 3 \
  --gradient_accumulation_steps 2 \
  --dataloader_num_workers 2 \
  --use_class_weights \
  --fp16 \
  --output_dir artifacts/deberta_politeness
```

Artifacts:
- HF model: `artifacts/deberta_politeness/hf_model`
- Metrics: `artifacts/deberta_politeness/metrics.json`

## 4) Hyperparameter tuning (recommended for >85%)

```bash
python -m tone_classifier.tune \
  --train_file data/train.csv \
  --validation_file data/valid.csv \
  --test_file data/test.csv \
  --text_column text \
  --label_column label \
  --profile high_accuracy \
  --max_experiments 24 \
  --fp16 \
  --use_class_weights \
  --base_output_dir artifacts/tuning
```

Outputs:
- per-experiment folders under `artifacts/tuning/exp_*`
- summary file: `artifacts/tuning/tuning_summary.json`
- best run file: `artifacts/tuning/best_experiment.json`

Pick the best run by highest `val_macro_f1` with strong `test_accuracy`.

## 5) Save Torch `.pt`

```bash
python -m tone_classifier.export_pt \
  --hf_model_dir artifacts/tuning/<best_exp_dir>/hf_model \
  --output_pt artifacts/model/politeness_deberta.pt
```

## 6) Inference

```bash
python -m tone_classifier.predict \
  --hf_model_dir artifacts/tuning/<best_exp_dir>/hf_model \
  --text "Can you please help me debug this issue when you have time?"
```

## 7) Practical notes for your 85% target

- Keep train/valid/test split stratified by label.
- Track both `accuracy` and `macro_f1` (neutral is often harder).
- If neutral recall is weak, keep `--use_class_weights` enabled.
- If GPU memory allows, try `batch_size=32` and fewer gradient steps.
- For runtime under 8 hours on one modern NVIDIA GPU, run the tuning grid once, then retrain best config with a fixed seed.

## 8) Download model from Google Drive to local machine

If your artifacts are on Drive and you want them on your laptop/desktop, zip first in Colab:

```bash
cd "$PROJECT_DIR"
zip -r artifacts_deberta.zip artifacts/deberta_politeness artifacts/model/politeness_deberta.pt
```

Then download from Colab:

```python
from google.colab import files
files.download("artifacts_deberta.zip")
```
