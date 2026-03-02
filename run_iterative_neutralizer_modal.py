#!/usr/bin/env python3
"""
Iterative neutralizer driven by Context-Cite attribution on Modal.

Workflow per example:
1) Run attribution / read current predicted tone
2) If tone is impolite or polite, remove the highest positive-contribution sentence
3) Re-run attribution on the edited text
4) Stop when prediction becomes neutral or max rounds reached
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run iterative neutralization using Modal Context-Cite attribution."
    )
    p.add_argument(
        "--input_batch_results",
        type=str,
        default="artifacts/attribution_batch_extreme_full/batch_results.jsonl",
        help="Path to existing batch_results.jsonl",
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Modal-accessible model path (e.g. /root/tone_classifier_outputs/.../hf_model)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/neutralizer",
        help="Output directory for neutralization results",
    )
    p.add_argument(
        "--target_labels",
        nargs="+",
        default=["impolite", "polite"],
        help="Only neutralize examples currently predicted as these labels",
    )
    p.add_argument(
        "--stop_label",
        type=str,
        default="neutral",
        help="Target label to stop neutralization",
    )
    p.add_argument("--max_rounds", type=int, default=5)
    p.add_argument("--num_ablations", type=int, default=256)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--context_keep_prob", type=float, default=0.8)
    p.add_argument("--min_context_sentences", type=int, default=1)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument(
        "--allow_non_positive_removal",
        action="store_true",
        help="If no positive contribution sentence exists, remove top score anyway.",
    )
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap for quick debugging",
    )
    return p.parse_args()


def _load_batch_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _split_into_sentences(text: str) -> List[str]:
    # Keep behavior close to tone_classifier.attribution.split_into_sentences.
    import re

    parts = re.split(r"([.!?]+)", text)
    result: List[str] = []
    cur = ""
    for part in parts:
        if re.match(r"[.!?]+", part):
            cur += part
            if cur.strip():
                result.append(cur.strip())
            cur = ""
        else:
            cur += part
    if cur.strip():
        result.append(cur.strip())
    return result if result else [text]


def _pick_sentence_to_remove(
    sentence_scores: List[Dict[str, Any]],
    allow_non_positive_removal: bool,
) -> Dict[str, Any] | None:
    if not sentence_scores:
        return None
    # sentence_scores from attribution are already sorted by attribution_score desc.
    for item in sentence_scores:
        if float(item.get("attribution_score", 0.0)) > 0.0:
            return item
    if allow_non_positive_removal:
        return sentence_scores[0]
    return None


def _remove_sentence_by_index(text: str, sent_idx: int) -> str:
    sents = _split_into_sentences(text)
    if sent_idx < 0 or sent_idx >= len(sents):
        return text
    kept = [s for i, s in enumerate(sents) if i != sent_idx]
    return " ".join(kept).strip()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_batch_results)
    if not input_path.exists():
        raise FileNotFoundError(f"Input batch_results not found: {input_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "neutralized_results.jsonl"
    out_summary = out_dir / "neutralizer_summary.json"

    records = _load_batch_results(input_path)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    target_labels = {x.strip().lower() for x in args.target_labels}
    stop_label = args.stop_label.strip().lower()

    from modal_app import app, run_attribution_analysis

    output_rows: List[Dict[str, Any]] = []
    with app.run():
        for ridx, rec in enumerate(records):
            text = str(rec.get("text", "")).strip()
            if not text:
                output_rows.append(
                    {
                        "index": ridx,
                        "status": "skipped_empty_text",
                        "original_text": text,
                    }
                )
                continue

            initial = rec.get("result", {}).get("baseline_prediction", {})
            initial_label = str(initial.get("label", "")).strip().lower()

            item_out: Dict[str, Any] = {
                "index": ridx,
                "true_label": rec.get("true_label"),
                "original_text": text,
                "initial_label": initial_label if initial_label else None,
                "rounds": [],
            }

            # If record has no usable initial label, compute once.
            if not initial_label:
                first_attr = run_attribution_analysis.remote(
                    model_path=args.model_path,
                    text=text,
                    num_ablations=args.num_ablations,
                    max_length=args.max_length,
                    context_keep_prob=args.context_keep_prob,
                    min_context_sentences=args.min_context_sentences,
                    random_seed=args.random_seed + ridx * 1000,
                )
                initial_label = (
                    str(first_attr.get("baseline_prediction", {}).get("label", ""))
                    .strip()
                    .lower()
                )
                item_out["initial_label"] = initial_label if initial_label else None

            if initial_label not in target_labels:
                item_out["status"] = "skipped_non_target_label"
                item_out["final_text"] = text
                item_out["final_label"] = initial_label if initial_label else None
                output_rows.append(item_out)
                continue

            cur_text = text
            cur_label = initial_label
            cur_probs = None
            status = "max_rounds_reached"

            for round_idx in range(1, args.max_rounds + 1):
                attr = run_attribution_analysis.remote(
                    model_path=args.model_path,
                    text=cur_text,
                    num_ablations=args.num_ablations,
                    max_length=args.max_length,
                    context_keep_prob=args.context_keep_prob,
                    min_context_sentences=args.min_context_sentences,
                    random_seed=args.random_seed + ridx * 1000 + round_idx,
                )
                pred = attr.get("baseline_prediction", {})
                cur_label = str(pred.get("label", "")).strip().lower()
                cur_probs = pred.get("probabilities")

                if cur_label == stop_label:
                    status = "success_neutralized"
                    item_out["rounds"].append(
                        {
                            "round": round_idx,
                            "label_before_removal": cur_label,
                            "probabilities_before_removal": cur_probs,
                            "action": "stop_already_neutral",
                        }
                    )
                    break

                sentence_scores = attr.get("sentences", [])
                chosen = _pick_sentence_to_remove(
                    sentence_scores=sentence_scores,
                    allow_non_positive_removal=args.allow_non_positive_removal,
                )
                if chosen is None:
                    status = "stopped_no_removable_sentence"
                    item_out["rounds"].append(
                        {
                            "round": round_idx,
                            "label_before_removal": cur_label,
                            "probabilities_before_removal": cur_probs,
                            "action": "stop_no_positive_contribution_sentence",
                        }
                    )
                    break

                remove_idx = int(chosen.get("index", -1))
                remove_score = float(chosen.get("attribution_score", 0.0))
                remove_sentence = str(chosen.get("sentence", ""))
                next_text = _remove_sentence_by_index(cur_text, remove_idx)
                if not next_text or next_text == cur_text:
                    status = "stopped_no_text_change"
                    item_out["rounds"].append(
                        {
                            "round": round_idx,
                            "label_before_removal": cur_label,
                            "probabilities_before_removal": cur_probs,
                            "action": "stop_no_text_change",
                            "chosen_index": remove_idx,
                            "chosen_score": remove_score,
                            "chosen_sentence": remove_sentence,
                        }
                    )
                    break

                item_out["rounds"].append(
                    {
                        "round": round_idx,
                        "label_before_removal": cur_label,
                        "probabilities_before_removal": cur_probs,
                        "chosen_index": remove_idx,
                        "chosen_score": remove_score,
                        "chosen_sentence": remove_sentence,
                        "text_after_removal": next_text,
                    }
                )
                cur_text = next_text

            # Final check after last modification if still not neutralized.
            if status == "max_rounds_reached":
                final_attr = run_attribution_analysis.remote(
                    model_path=args.model_path,
                    text=cur_text,
                    num_ablations=args.num_ablations,
                    max_length=args.max_length,
                    context_keep_prob=args.context_keep_prob,
                    min_context_sentences=args.min_context_sentences,
                    random_seed=args.random_seed + ridx * 1000 + args.max_rounds + 100,
                )
                final_pred = final_attr.get("baseline_prediction", {})
                final_label = str(final_pred.get("label", "")).strip().lower()
                if final_label == stop_label:
                    status = "success_neutralized"
                cur_label = final_label
                cur_probs = final_pred.get("probabilities")

            item_out["status"] = status
            item_out["final_text"] = cur_text
            item_out["final_label"] = cur_label if cur_label else None
            item_out["final_probabilities"] = cur_probs
            output_rows.append(item_out)

            if (ridx + 1) % 20 == 0:
                print(f"Processed {ridx + 1}/{len(records)} examples")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(output_rows)
    attempted = sum(1 for r in output_rows if str(r.get("status", "")).startswith("success") or str(r.get("status", "")).startswith("max_rounds") or str(r.get("status", "")).startswith("stopped"))
    success = sum(1 for r in output_rows if r.get("status") == "success_neutralized")
    skipped_non_target = sum(1 for r in output_rows if r.get("status") == "skipped_non_target_label")
    summary = {
        "total_examples": total,
        "attempted_examples": attempted,
        "successful_neutralizations": success,
        "success_rate_over_attempted": (success / attempted) if attempted else 0.0,
        "skipped_non_target_label": skipped_non_target,
        "input_batch_results": str(input_path),
        "model_path": args.model_path,
        "target_labels": sorted(target_labels),
        "stop_label": stop_label,
        "max_rounds": args.max_rounds,
        "num_ablations": args.num_ablations,
    }
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved neutralized rows: {out_jsonl}")
    print(f"Saved summary: {out_summary}")
    print(
        "Neutralization done: "
        f"{success}/{attempted} succeeded ({summary['success_rate_over_attempted']:.2%})."
    )


if __name__ == "__main__":
    main()

