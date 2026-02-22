from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

from tone_classifier.data import (
    DataConfig,
    ID_TO_LABEL,
    LABEL_TO_ID,
    class_weights_from_dataset,
    load_politeness_dataset,
    prepare_dataset,
)
from tone_classifier.modeling import WeightedLossTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RoBERTa politeness classifier")
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--validation_file", type=str, default=None)
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--label_column", type=str, default="label")

    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--max_length", type=int, default=128)

    p.add_argument("--output_dir", type=str, default="artifacts/roberta_politeness")
    p.add_argument("--num_train_epochs", type=float, default=5.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_class_weights", action="store_true")

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=50)
    return p.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision_impolite": float(precision[0]),
        "precision_neutral": float(precision[1]),
        "precision_polite": float(precision[2]),
        "recall_impolite": float(recall[0]),
        "recall_neutral": float(recall[1]),
        "recall_polite": float(recall[2]),
        "f1_impolite": float(f1_per_class[0]),
        "f1_neutral": float(f1_per_class[1]),
        "f1_polite": float(f1_per_class[2]),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        test_file=args.test_file,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    raw_dataset = load_politeness_dataset(data_cfg)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenized_dataset = prepare_dataset(
        raw_dataset,
        tokenizer=tokenizer,
        text_column=args.text_column,
        label_column=args.label_column,
        max_length=args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    use_fp16 = args.fp16
    if not use_fp16 and torch.cuda.is_available():
        use_fp16 = True

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        fp16=use_fp16,
        bf16=args.bf16,
        dataloader_num_workers=4,
        remove_unused_columns=True,
    )

    weights = class_weights_from_dataset(tokenized_dataset) if args.use_class_weights else None

    trainer = WeightedLossTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "hf_model"))
    tokenizer.save_pretrained(str(output_dir / "hf_model"))

    val_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["validation"])
    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"], metric_key_prefix="test")

    metrics = {
        "train": train_result.metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "class_weights": list(weights) if weights is not None else None,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Validation accuracy:", round(val_metrics.get("eval_accuracy", 0.0), 4))
    print("Validation macro_f1:", round(val_metrics.get("eval_macro_f1", 0.0), 4))
    print("Test accuracy:", round(test_metrics.get("test_accuracy", 0.0), 4))
    print("Test macro_f1:", round(test_metrics.get("test_macro_f1", 0.0), 4))
    print("Saved model to:", output_dir / "hf_model")


if __name__ == "__main__":
    main()
