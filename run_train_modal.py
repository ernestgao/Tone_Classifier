#!/usr/bin/env python3
"""
Trigger tone-classifier training on Modal GPUs.

Outputs:
- local summary json with metrics and remote artifact paths
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _print_per_class_precision_recall(split_name: str, split_metrics: dict, prefix: str) -> None:
    labels = ["impolite", "neutral", "polite"]
    print(f"{split_name} per-class metrics:")
    for label in labels:
        precision_key = f"{prefix}precision_{label}"
        recall_key = f"{prefix}recall_{label}"
        precision = split_metrics.get(precision_key)
        recall = split_metrics.get(recall_key)
        print(f"  {label}: precision={precision}, recall={recall}")


def _apply_large_preset_defaults(args: argparse.Namespace) -> None:
    """
    Keep large preset robust even when CLI uses base defaults.
    """
    # If user did not tune these away from base defaults, switch to safer large-model defaults.
    if args.per_device_train_batch_size == 16:
        args.per_device_train_batch_size = 8
    if args.per_device_eval_batch_size == 32:
        args.per_device_eval_batch_size = 16
    if args.gradient_accumulation_steps == 2:
        args.gradient_accumulation_steps = 4
    if abs(args.learning_rate - 2e-5) < 1e-12:
        args.learning_rate = 1.5e-5
    if args.output_subdir == "deberta_modal_train":
        args.output_subdir = "deberta_large_modal_train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tone classifier training on Modal."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="HF model id for classifier fine-tuning",
    )
    parser.add_argument("--train_file", type=str, default="data/train.csv")
    parser.add_argument("--validation_file", type=str, default="data/valid.csv")
    parser.add_argument("--test_file", type=str, default="data/test.csv")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--output_subdir", type=str, default="deberta_modal_train")
    parser.add_argument("--num_train_epochs", type=float, default=8.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--use_large_preset",
        action="store_true",
        help="Use larger-model preset function (H100 + larger defaults).",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default="artifacts/modal_train/run_summary.json",
        help="Local path to save training run summary json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_large_preset:
        _apply_large_preset_defaults(args)

    from modal_app import app, run_modal_training, run_modal_training_large

    call_args = dict(
        model_name=args.model_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        test_file=args.test_file,
        text_column=args.text_column,
        label_column=args.label_column,
        output_subdir=args.output_subdir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        max_length=args.max_length,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        use_class_weights=args.use_class_weights,
        fp16=args.fp16,
    )

    print("Submitting Modal training job...")
    with app.run():
        try:
            if args.use_large_preset:
                result = run_modal_training_large.remote(**call_args)
            else:
                result = run_modal_training.remote(**call_args)
        except Exception as e:
            raise RuntimeError(
                "Modal training request failed. "
                "Check remote traceback above. If it mentions OOM, reduce batch sizes "
                "(e.g., train=4, eval=8) or lower max_length."
            ) from e

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    metrics = result.get("metrics", {})
    val = metrics.get("validation", {})
    test = metrics.get("test", {})
    print(f"Saved run summary: {summary_path}")
    print(f"Modal output_dir: {result.get('output_dir')}")
    print(f"Modal hf_model_dir: {result.get('hf_model_dir')}")
    print(f"Val acc: {val.get('eval_accuracy')}, Val macro_f1: {val.get('eval_macro_f1')}")
    _print_per_class_precision_recall("Val", val, "eval_")
    print(f"Test acc: {test.get('test_accuracy')}, Test macro_f1: {test.get('test_macro_f1')}")
    _print_per_class_precision_recall("Test", test, "test_")


if __name__ == "__main__":
    main()

