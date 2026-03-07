# Politeness Tone Classifier

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

### Attention-based attribution (last-layer `[CLS]` focus)

Use this to locate tone-related words/phrases by:
- taking `[CLS] -> token` attention from the last layer
- for each token, expanding only to the right: `w`, `w w+1`, `w w+1 w+2`, ...
- scoring each candidate phrase by average attention over its tokens
- stopping expansion early using score-ratio criteria
- globally ranking all single words/phrases and returning top-k

```bash
python -m tone_classifier.predict \
  --hf_model_dir artifacts/tuning/<best_exp_dir>/hf_model \
  --text "Can you please help me debug this issue when you have time?" \
  --show_attribution \
  --attribution_top_k 5 \
  --attribution_max_phrase_tokens 6 \
  --attribution_max_overlap_ratio 0.8 \
  --attribution_drop_vs_initial_threshold 0.75 \
  --attribution_small_drop_no_change_threshold 0.90 \
  --attribution_drop_vs_prev_threshold 0.80 \
  --attribution_content_words_only
```

Iterative erasure mode (repeat attribution after masking standout tokens):

```bash
python -m tone_classifier.predict \
  --hf_model_dir artifacts/tuning/<best_exp_dir>/hf_model \
  --text "Why are you stupid? You are the most retarded agent I have ever seen. I hate you" \
  --show_attribution \
  --attribution_top_k 10 \
  --attribution_max_phrase_tokens 1 \
  --attribution_content_words_only \
  --attribution_iterative_erasure \
  --attribution_iter_max_rounds 5 \
  --attribution_iter_median_ratio 1.8 \
  --attribution_iter_max_ratio 0.8 \
  --attribution_iter_min_token_score 0.02 \
  --attribution_iter_eval_top_n 8 \
  --attribution_iter_remove_top_n 1 \
  --attribution_iter_min_prob_drop 0.00
```

Use deletion-style erasure instead of `[MASK]` replacement:

```bash
python -m tone_classifier.predict \
  ... \
  --attribution_iterative_erasure \
  --attribution_iter_mask_token ""
```

### Top-k masking on original prompt + MLM neutral substitutions

Mask the top-k attributed words (or phrases) directly on the original prompt:

```bash
python -m tone_classifier.predict \
  --hf_model_dir artifacts/tuning/<best_exp_dir>/hf_model \
  --text "Why are you stupid? You are the most retarded agent I have ever seen. I hate you" \
  --show_attribution \
  --attribution_top_k 10 \
  --attribution_max_phrase_tokens 1 \
  --attribution_content_words_only \
  --attribution_mask_top_k 5
```

If you have a fine-tuned BERT-style MLM, fill those masks with constrained reranking
(neutral probability + token similarity + sentence similarity):

```bash
python -m tone_classifier.predict \
  --hf_model_dir artifacts/tuning/<best_exp_dir>/hf_model \
  --text "Why are you stupid? You are the most retarded agent I have ever seen. I hate you" \
  --show_attribution \
  --attribution_top_k 10 \
  --attribution_max_phrase_tokens 1 \
  --attribution_content_words_only \
  --attribution_mask_top_k 3 \
  --fill_masks_with_mlm \
  --mlm_model_dir artifacts/neutral_mlm_paradetox/hf_model \
  --mlm_target_label neutral \
  --mlm_rerank_top_k 20 \
  --mlm_neutral_weight 1.0 \
  --mlm_token_similarity_weight 0.9 \
  --mlm_sentence_similarity_weight 0.6 \
  --mlm_min_token_cosine 0.55
```

The script prints:
- selected top-k spans from attribution
- `mask_targets_in_order` (original words/phrases being replaced)
- `masked_text` with `[MASK]` tokens
- per-mask fill steps including `target_prob`, `token_cos`, `sent_cos`, `combined`
- final filled text + classifier probabilities

Tuning tips:
- Increase `--mlm_min_token_cosine` (e.g. `0.6`) for stronger meaning preservation.
- Increase `--mlm_neutral_weight` if outputs are still too toxic.
- Lower `--attribution_mask_top_k` if rewriting becomes too aggressive.

### Train a neutral-domain MLM (BERT-style)

Fine-tune MLM on your neutral text so mask filling prefers neutral substitutions:

```bash
python -m tone_classifier.train_mlm \
  --train_file data/train.csv \
  --validation_file data/valid.csv \
  --text_column text \
  --label_column label \
  --filter_to_neutral \
  --neutral_label_values neutral 0 \
  --model_name bert-base-uncased \
  --max_length 128 \
  --mlm_probability 0.15 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --output_dir artifacts/neutral_mlm
```

## 7) Download model from Google Drive to local machine

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
