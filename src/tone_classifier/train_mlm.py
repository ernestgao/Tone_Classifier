from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a masked language model (MLM), optionally on neutral-only text."
    )
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--validation_file", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--label_column", type=str, default="label")
    p.add_argument("--filter_to_neutral", action="store_true")
    p.add_argument(
        "--neutral_label_values",
        nargs="+",
        default=["neutral", "0"],
        help="Accepted neutral label values when --filter_to_neutral is enabled.",
    )
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--output_dir", type=str, default="artifacts/neutral_mlm")
    p.add_argument("--num_train_epochs", type=float, default=4.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    return p.parse_args()


def _dataset_loader_name(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(
        f"Unsupported file extension '{suffix}' for {path}. "
        "Use csv/json/jsonl/parquet."
    )


def _load_text_dataset(args: argparse.Namespace) -> DatasetDict:
    loader_name = _dataset_loader_name(args.train_file)
    data_files: dict[str, str] = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file
    return load_dataset(loader_name, data_files=data_files)


def _normalize_label(v: Any) -> str:
    return str(v).strip().lower()


def _filter_dataset_to_neutral(dataset: DatasetDict, args: argparse.Namespace) -> DatasetDict:
    neutral_values = {_normalize_label(v) for v in args.neutral_label_values}

    def is_neutral(example: dict[str, Any]) -> bool:
        return _normalize_label(example.get(args.label_column, "")) in neutral_values

    for split_name in dataset.keys():
        if args.label_column not in dataset[split_name].column_names:
            raise ValueError(
                f"--filter_to_neutral requires label column '{args.label_column}' "
                f"in split '{split_name}'."
            )
    return dataset.filter(is_neutral)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = _load_text_dataset(args)
    if args.filter_to_neutral:
        raw_dataset = _filter_dataset_to_neutral(raw_dataset, args)

    if "train" not in raw_dataset:
        raise ValueError("Dataset must contain a train split.")
    if len(raw_dataset["train"]) == 0:
        raise ValueError("Train split is empty after filtering.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.mask_token is None:
        raise ValueError(
            f"Tokenizer for {args.model_name} has no mask token. "
            "Choose a masked language model backbone."
        )

    def tokenize_batch(examples: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [str(x) for x in examples[args.text_column]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True,
        )

    remove_columns = raw_dataset["train"].column_names
    tokenized = raw_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=remove_columns,
    )

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )

    use_fp16 = args.fp16
    if not use_fp16 and torch.cuda.is_available():
        use_fp16 = True

    has_validation = "validation" in tokenized and len(tokenized["validation"]) > 0

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps" if has_validation else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        report_to="none",
        fp16=use_fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if has_validation else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "hf_model"))
    tokenizer.save_pretrained(str(output_dir / "hf_model"))

    metrics: dict[str, Any] = {"train": train_result.metrics}
    if has_validation:
        eval_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
        eval_loss = float(eval_metrics.get("eval_loss", 0.0))
        eval_perplexity = math.exp(eval_loss) if eval_loss < 50 else float("inf")
        eval_metrics["eval_perplexity"] = float(eval_perplexity)
        metrics["validation"] = eval_metrics

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved MLM model to:", output_dir / "hf_model")
    if "validation" in metrics:
        print("Validation loss:", round(metrics["validation"].get("eval_loss", 0.0), 4))
        print(
            "Validation perplexity:",
            round(metrics["validation"].get("eval_perplexity", 0.0), 4),
        )


if __name__ == "__main__":
    main()
