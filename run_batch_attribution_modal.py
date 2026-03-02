#!/usr/bin/env python3
"""
Batch attribution runner using Modal GPU functions.

This script:
1) loads texts from CSV/JSON/JSONL/TXT
2) sends them to Modal in chunks
3) saves per-example outputs and a summary report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch attribution on Modal with larger models if needed."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file path (.csv/.json/.jsonl/.txt)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Text column for csv/json/jsonl inputs",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Optional label column for accuracy/F1 (csv/json/jsonl).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HF model path/id for AutoModelForSequenceClassification",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/attribution_batch",
        help="Output directory",
    )
    parser.add_argument(
        "--num_ablations",
        type=int,
        default=512,
        help="Number of sampled ablations per example",
    )
    parser.add_argument(
        "--context_keep_prob",
        type=float,
        default=0.8,
        help="Probability of keeping each non-target sentence",
    )
    parser.add_argument(
        "--min_context_sentences",
        type=int,
        default=1,
        help="Minimum context sentence count in each sampled ablation",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max token length",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8,
        help="How many examples per Modal remote call",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of input examples",
    )
    parser.add_argument(
        "--use_large_model",
        action="store_true",
        help="Use large-model Modal function (2xH100 path)",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use 8-bit quantization in large-model path",
    )
    return parser.parse_args()


def normalize_label(raw: Any) -> Optional[str]:
    """
    Normalize various tone labels to one of:
    - impolite
    - neutral
    - polite
    Returns None if missing/unrecognized.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    mapping = {
        "-1": "impolite",
        "0": "neutral",
        "1": "polite",
        "impolite": "impolite",
        "rude": "impolite",
        "neutral": "neutral",
        "polite": "polite",
    }
    return mapping.get(s)


def _load_from_csv(path: Path, text_column: str, label_column: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = str(row.get(text_column, "")).strip()
            if text:
                rows.append(
                    {
                        "text": text,
                        "true_label": normalize_label(row.get(label_column)),
                    }
                )
    return rows


def _load_from_json(path: Path, text_column: str, label_column: str) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                text = str(item.get(text_column, "")).strip()
                if text:
                    rows.append(
                        {
                            "text": text,
                            "true_label": normalize_label(item.get(label_column)),
                        }
                    )
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    rows.append({"text": text, "true_label": None})
    elif isinstance(data, dict):
        if text_column in data and isinstance(data[text_column], list):
            label_list = data.get(label_column, [])
            if not isinstance(label_list, list):
                label_list = []
            for idx, item in enumerate(data[text_column]):
                text = str(item).strip()
                if not text:
                    continue
                raw_label = label_list[idx] if idx < len(label_list) else None
                rows.append({"text": text, "true_label": normalize_label(raw_label)})
    return rows


def _load_from_jsonl(path: Path, text_column: str, label_column: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                text = str(item.get(text_column, "")).strip()
                if text:
                    rows.append(
                        {
                            "text": text,
                            "true_label": normalize_label(item.get(label_column)),
                        }
                    )
    return rows


def _load_from_txt(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append({"text": text, "true_label": None})
    return rows


def load_examples(path: Path, text_column: str, label_column: str) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_from_csv(path, text_column, label_column)
    if suffix == ".json":
        return _load_from_json(path, text_column, label_column)
    if suffix == ".jsonl":
        return _load_from_jsonl(path, text_column, label_column)
    if suffix == ".txt":
        return _load_from_txt(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def resolve_model_path_for_modal(model_path: str) -> str:
    """
    Resolve model identifier for Modal runtime.

    This script runs attribution remotely on Modal. A local filesystem path
    (e.g., .\\artifacts\\... on Windows) is not visible inside Modal containers.
    Users should pass either:
    - a Hugging Face model id, or
    - a Modal-accessible path, e.g. /root/tone_classifier_outputs/<subdir>/hf_model
    """
    local_path = Path(model_path)
    if local_path.exists():
        raise ValueError(
            "Detected local model path, but this script runs on Modal remote workers.\n"
            f"Local path: {local_path}\n"
            "Use a Modal path like '/root/tone_classifier_outputs/<subdir>/hf_model' "
            "(from run_train_modal.py output 'Modal hf_model_dir'), "
            "or use a Hugging Face model id."
        )

    # Normalize Windows-style separators for remote use.
    normalized = model_path.replace("\\", "/")

    # Common accidental local pattern that would be misread as HF repo id.
    if normalized.startswith("./") or normalized.startswith(".\\"):
        raise ValueError(
            "Relative local path provided for --model_path, which Modal cannot access.\n"
            f"Provided: {model_path}\n"
            "Use '/root/tone_classifier_outputs/<subdir>/hf_model' or a Hugging Face model id."
        )

    return normalized


def chunk_list(items: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def compute_classification_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute accuracy and macro-F1 on examples that have true labels.
    """
    labels = ["impolite", "neutral", "polite"]
    supported = set(labels)

    eval_records = []
    for rec in records:
        if not rec.get("ok"):
            continue
        true_label = normalize_label(rec.get("true_label"))
        pred_label = normalize_label(
            rec.get("result", {}).get("baseline_prediction", {}).get("label")
        )
        if true_label in supported and pred_label in supported:
            eval_records.append((true_label, pred_label))

    if not eval_records:
        return {
            "labeled_examples": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_label": {},
            "confusion_matrix": {},
        }

    total = len(eval_records)
    correct = sum(1 for t, p in eval_records if t == p)
    accuracy = correct / total

    confusion: Dict[str, Dict[str, int]] = {
        t: {p: 0 for p in labels} for t in labels
    }
    for t, p in eval_records:
        confusion[t][p] += 1

    per_label: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in labels if t != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }
        f1_scores.append(f1)

    return {
        "labeled_examples": total,
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def summarize_results(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_records = [r for r in records if r.get("ok")]
    fail_records = [r for r in records if not r.get("ok")]

    label_counts: Dict[str, int] = {}
    avg_top_score = []
    avg_elapsed = []

    for rec in ok_records:
        avg_elapsed.append(float(rec.get("elapsed_seconds", 0.0)))
        result = rec.get("result", {})
        baseline = result.get("baseline_prediction", {})
        label = baseline.get("label")
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
        sentences = result.get("sentences", [])
        if sentences:
            avg_top_score.append(float(sentences[0].get("attribution_score", 0.0)))

    summary = {
        "total_examples": len(records),
        "ok_examples": len(ok_records),
        "failed_examples": len(fail_records),
        "ok_rate": (len(ok_records) / len(records)) if records else 0.0,
        "baseline_label_counts": label_counts,
        "mean_top_sentence_score": (sum(avg_top_score) / len(avg_top_score)) if avg_top_score else 0.0,
        "mean_elapsed_seconds": (sum(avg_elapsed) / len(avg_elapsed)) if avg_elapsed else 0.0,
    }
    if fail_records:
        summary["sample_errors"] = [f.get("error", "unknown") for f in fail_records[:5]]

    cls_metrics = compute_classification_metrics(records)
    summary.update(cls_metrics)
    return summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_path = resolve_model_path_for_modal(args.model_path)

    examples = load_examples(input_path, args.text_column, args.label_column)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    if not examples:
        raise ValueError("No valid input texts found.")

    texts = [ex["text"] for ex in examples]
    print(f"Loaded {len(texts)} texts from {input_path}")
    chunks = chunk_list(examples, args.chunk_size)
    print(f"Running {len(chunks)} chunk(s) on Modal (chunk_size={args.chunk_size})")

    from modal_app import (
        app,
        run_batch_attribution_analysis,
        run_large_model_batch_attribution,
    )

    all_records: List[Dict[str, Any]] = []
    with app.run():
        offset = 0
        for chunk_id, chunk in enumerate(chunks):
            print(f"[Chunk {chunk_id + 1}/{len(chunks)}] size={len(chunk)}")
            chunk_texts = [ex["text"] for ex in chunk]

            if args.use_large_model:
                remote_output = run_large_model_batch_attribution.remote(
                    model_name=resolved_model_path,
                    texts=chunk_texts,
                    num_ablations=args.num_ablations,
                    max_length=args.max_length,
                    use_quantization=args.use_quantization,
                    context_keep_prob=args.context_keep_prob,
                    min_context_sentences=args.min_context_sentences,
                    random_seed=args.random_seed + offset,
                )
            else:
                remote_output = run_batch_attribution_analysis.remote(
                    model_path=resolved_model_path,
                    texts=chunk_texts,
                    num_ablations=args.num_ablations,
                    max_length=args.max_length,
                    context_keep_prob=args.context_keep_prob,
                    min_context_sentences=args.min_context_sentences,
                    random_seed=args.random_seed + offset,
                )

            batch_results = remote_output.get("batch_results", [])
            for rec in batch_results:
                local_index = int(rec.get("index", 0))
                global_index = offset + local_index
                rec["global_index"] = global_index
                if 0 <= global_index < len(examples):
                    rec["true_label"] = examples[global_index].get("true_label")
                all_records.append(rec)

            offset += len(chunk)

    # Stable order
    all_records.sort(key=lambda x: int(x.get("global_index", 0)))

    results_jsonl = output_dir / "batch_results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize_results(all_records)
    summary.update(
        {
            "input_file": str(input_path),
            "model_path": resolved_model_path,
            "use_large_model": args.use_large_model,
            "use_quantization": args.use_quantization,
            "num_ablations": args.num_ablations,
            "context_keep_prob": args.context_keep_prob,
            "min_context_sentences": args.min_context_sentences,
            "max_length": args.max_length,
            "chunk_size": args.chunk_size,
            "label_column": args.label_column,
        }
    )

    summary_json = output_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved per-example results: {results_jsonl}")
    print(f"Saved summary: {summary_json}")
    print(
        "Done: "
        f"{summary['ok_examples']}/{summary['total_examples']} succeeded "
        f"({summary['ok_rate']:.2%})."
    )


if __name__ == "__main__":
    main()

