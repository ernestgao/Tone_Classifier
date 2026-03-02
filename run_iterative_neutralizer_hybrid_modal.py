#!/usr/bin/env python3
"""
Hybrid iterative neutralizer on Modal.

Policy:
- If comma-based clause count >= threshold, use sentence-level removal.
- Otherwise, use token-level removal based on attention token importance.

This script is added as a comparison group and does NOT modify the existing
run_iterative_neutralizer_modal.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid neutralizer: sentence-level for long texts, token-level for short texts."
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
        default="artifacts/neutralizer_hybrid",
        help="Output directory",
    )
    p.add_argument(
        "--target_labels",
        nargs="+",
        default=["impolite", "polite"],
        help="Only neutralize examples currently predicted as these labels",
    )
    p.add_argument("--stop_label", type=str, default="neutral")
    p.add_argument("--max_rounds", type=int, default=5)
    p.add_argument("--num_ablations", type=int, default=256)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--context_keep_prob", type=float, default=0.8)
    p.add_argument("--min_context_sentences", type=int, default=1)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument(
        "--clause_threshold",
        type=int,
        default=4,
        help="Use sentence-level removal if comma-based clause count >= this value.",
    )
    p.add_argument(
        "--allow_non_positive_sentence_removal",
        action="store_true",
        help="If no positive-contribution sentence exists, still remove top sentence.",
    )
    p.add_argument(
        "--token_min_chars",
        type=int,
        default=2,
        help="Minimum cleaned token length for token-level deletion.",
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
            if line:
                rows.append(json.loads(line))
    return rows


def _split_into_sentences(text: str) -> List[str]:
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


def _comma_clause_count(text: str) -> int:
    # User rule: comma also counts as a sentence-like unit.
    chunks = [x.strip() for x in re.split(r"[,，]", text) if x.strip()]
    return max(1, len(chunks))


def _pick_sentence_to_remove(
    sentence_scores: List[Dict[str, Any]],
    allow_non_positive_removal: bool,
) -> Dict[str, Any] | None:
    if not sentence_scores:
        return None
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


def _clean_token(tok: str) -> str:
    t = tok.strip()
    # RoBERTa-style word boundary marker
    if t.startswith("Ġ"):
        t = t[1:]
    # Common special tokens
    if t.startswith("<") and t.endswith(">"):
        return ""
    # Keep alnum, apostrophe, hyphen only
    t = re.sub(r"[^A-Za-z0-9'\-]", "", t)
    return t


def _pick_token_to_remove(
    text: str,
    tokens: List[str],
    token_importance: List[float],
    min_chars: int,
) -> Tuple[str | None, float | None]:
    pairs = list(zip(tokens, token_importance))
    pairs.sort(key=lambda x: float(x[1]), reverse=True)
    for raw_tok, score in pairs:
        tok = _clean_token(str(raw_tok))
        if len(tok) < min_chars:
            continue
        if tok.lower() in {"s", "t", "re", "ve", "ll", "d", "m"}:
            continue
        if re.search(re.escape(tok), text, flags=re.IGNORECASE):
            return tok, float(score)
    return None, None


def _remove_token_once(text: str, token: str) -> str:
    # Remove first occurrence, case-insensitive.
    out, n = re.subn(re.escape(token), "", text, count=1, flags=re.IGNORECASE)
    if n <= 0:
        return text
    # Normalize spaces around punctuation.
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    return out


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

    from modal_app import app, run_attribution_analysis, run_attention_attribution

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

            initial_pred = rec.get("result", {}).get("baseline_prediction", {})
            initial_label = str(initial_pred.get("label", "")).strip().lower()
            item_out: Dict[str, Any] = {
                "index": ridx,
                "true_label": rec.get("true_label"),
                "original_text": text,
                "initial_label": initial_label if initial_label else None,
                "rounds": [],
            }

            if initial_label not in target_labels:
                item_out["status"] = "skipped_non_target_label"
                item_out["final_text"] = text
                item_out["final_label"] = initial_label if initial_label else None
                output_rows.append(item_out)
                continue

            cur_text = text
            cur_label = initial_label
            cur_probs = initial_pred.get("probabilities")
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
                            "label_before_edit": cur_label,
                            "probabilities_before_edit": cur_probs,
                            "action": "stop_already_neutral",
                        }
                    )
                    break

                clause_count = _comma_clause_count(cur_text)
                use_sentence_level = clause_count >= args.clause_threshold

                if use_sentence_level:
                    sentence_scores = attr.get("sentences", [])
                    chosen = _pick_sentence_to_remove(
                        sentence_scores=sentence_scores,
                        allow_non_positive_removal=args.allow_non_positive_sentence_removal,
                    )
                    if chosen is None:
                        status = "stopped_no_removable_sentence"
                        item_out["rounds"].append(
                            {
                                "round": round_idx,
                                "method": "sentence",
                                "comma_clause_count": clause_count,
                                "label_before_edit": cur_label,
                                "probabilities_before_edit": cur_probs,
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
                                "method": "sentence",
                                "comma_clause_count": clause_count,
                                "label_before_edit": cur_label,
                                "probabilities_before_edit": cur_probs,
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
                            "method": "sentence",
                            "comma_clause_count": clause_count,
                            "label_before_edit": cur_label,
                            "probabilities_before_edit": cur_probs,
                            "chosen_index": remove_idx,
                            "chosen_score": remove_score,
                            "chosen_sentence": remove_sentence,
                            "text_after_edit": next_text,
                        }
                    )
                    cur_text = next_text
                    continue

                # Token-level path for short comma-based texts.
                att = run_attention_attribution.remote(
                    model_path=args.model_path,
                    text=cur_text,
                    aggregation_method="mean",
                    max_length=args.max_length,
                )
                tokens = att.get("tokens", [])
                token_importance = att.get("token_importance", [])
                token, token_score = _pick_token_to_remove(
                    text=cur_text,
                    tokens=tokens,
                    token_importance=token_importance,
                    min_chars=args.token_min_chars,
                )
                if token is None:
                    status = "stopped_no_removable_token"
                    item_out["rounds"].append(
                        {
                            "round": round_idx,
                            "method": "token",
                            "comma_clause_count": clause_count,
                            "label_before_edit": cur_label,
                            "probabilities_before_edit": cur_probs,
                            "action": "stop_no_removable_token",
                        }
                    )
                    break

                next_text = _remove_token_once(cur_text, token)
                if not next_text or next_text == cur_text:
                    status = "stopped_no_text_change"
                    item_out["rounds"].append(
                        {
                            "round": round_idx,
                            "method": "token",
                            "comma_clause_count": clause_count,
                            "label_before_edit": cur_label,
                            "probabilities_before_edit": cur_probs,
                            "action": "stop_no_text_change",
                            "chosen_token": token,
                            "chosen_token_score": token_score,
                        }
                    )
                    break

                item_out["rounds"].append(
                    {
                        "round": round_idx,
                        "method": "token",
                        "comma_clause_count": clause_count,
                        "label_before_edit": cur_label,
                        "probabilities_before_edit": cur_probs,
                        "chosen_token": token,
                        "chosen_token_score": token_score,
                        "text_after_edit": next_text,
                    }
                )
                cur_text = next_text

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
    attempted = sum(
        1
        for r in output_rows
        if str(r.get("status", "")).startswith("success")
        or str(r.get("status", "")).startswith("max_rounds")
        or str(r.get("status", "")).startswith("stopped")
    )
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
        "clause_threshold": args.clause_threshold,
        "policy": "comma_clause_count>=threshold -> sentence-level else token-level",
    }
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved neutralized rows: {out_jsonl}")
    print(f"Saved summary: {out_summary}")
    print(
        "Hybrid neutralization done: "
        f"{success}/{attempted} succeeded ({summary['success_rate_over_attempted']:.2%})."
    )


if __name__ == "__main__":
    main()

