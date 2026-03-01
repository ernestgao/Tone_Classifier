from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tone_classifier.attribution import extract_cls_attention_attribution


ID_TO_LABEL = {
    0: "impolite",
    1: "neutral",
    2: "polite",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict politeness label")
    p.add_argument("--hf_model_dir", type=str, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--show_attribution", action="store_true")
    p.add_argument("--attribution_top_k", type=int, default=5)
    p.add_argument("--attribution_neighbor_threshold", type=float, default=0.01)
    p.add_argument("--attribution_max_phrase_tokens", type=int, default=6)
    p.add_argument("--attribution_max_overlap_ratio", type=float, default=0.8)
    p.add_argument("--attribution_drop_vs_initial_threshold", type=float, default=0.75)
    p.add_argument("--attribution_small_drop_no_change_threshold", type=float, default=0.90)
    p.add_argument("--attribution_drop_vs_prev_threshold", type=float, default=0.80)
    p.add_argument("--attribution_content_words_only", action="store_true")
    p.add_argument("--attribution_min_alpha_chars", type=int, default=3)
    p.add_argument("--attribution_disable_dedup_by_text", action="store_true")
    p.add_argument("--attribution_iterative_erasure", action="store_true")
    p.add_argument("--attribution_iter_max_rounds", type=int, default=5)
    p.add_argument("--attribution_iter_median_ratio", type=float, default=1.8)
    p.add_argument("--attribution_iter_max_ratio", type=float, default=0.8)
    p.add_argument("--attribution_iter_min_token_score", type=float, default=0.02)
    p.add_argument("--attribution_iter_eval_top_n", type=int, default=8)
    p.add_argument("--attribution_iter_remove_top_n", type=int, default=1)
    p.add_argument("--attribution_iter_min_prob_drop", type=float, default=0.0)
    p.add_argument("--attribution_iter_mask_token", type=str, default=None)
    return p.parse_args()


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)


def _run_single_prediction(
    *,
    args: argparse.Namespace,
    text: str,
    tokenizer: Any,
    model: Any,
    device: str,
    with_attribution: bool | None = None,
) -> dict[str, Any]:
    use_attribution = args.show_attribution if with_attribution is None else with_attribution
    with torch.no_grad():
        tokenizer_kwargs = {
            "truncation": True,
            "max_length": args.max_length,
            "return_tensors": "pt",
        }
        if use_attribution and tokenizer.is_fast:
            tokenizer_kwargs["return_offsets_mapping"] = True

        inputs = tokenizer(text, **tokenizer_kwargs)
        offsets = None
        if use_attribution and "offset_mapping" in inputs:
            offsets = [tuple(pair) for pair in inputs["offset_mapping"][0].tolist()]
            del inputs["offset_mapping"]

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_attentions=use_attribution, return_dict=True)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred = int(torch.argmax(probs).item())

    attributions: list[dict[str, Any]] | None = None
    if use_attribution and outputs.attentions is not None:
        attributions = extract_cls_attention_attribution(
            tokenizer=tokenizer,
            text=text,
            input_ids=inputs["input_ids"][0].detach().cpu(),
            attention_mask=inputs["attention_mask"][0].detach().cpu(),
            last_layer_attention=outputs.attentions[-1][0].detach().cpu(),
            top_k=args.attribution_top_k,
            neighbor_threshold=args.attribution_neighbor_threshold,
            max_phrase_tokens=args.attribution_max_phrase_tokens,
            max_overlap_ratio=args.attribution_max_overlap_ratio,
            drop_vs_initial_threshold=args.attribution_drop_vs_initial_threshold,
            small_drop_no_change_threshold=args.attribution_small_drop_no_change_threshold,
            drop_vs_prev_threshold=args.attribution_drop_vs_prev_threshold,
            content_words_only=args.attribution_content_words_only,
            dedup_by_text=not args.attribution_disable_dedup_by_text,
            min_alpha_chars=args.attribution_min_alpha_chars,
            offsets=offsets,
        )

    return {
        "text": text,
        "pred": pred,
        "probs": probs.detach().cpu(),
        "attributions": attributions,
    }


def _print_probabilities(probs: torch.Tensor) -> None:
    print("probabilities:")
    for i, prob in enumerate(probs.tolist()):
        print(f"  {ID_TO_LABEL[i]}: {prob:.4f}")


def _print_attributions(attributions: list[dict[str, Any]] | None) -> None:
    print("\nattention_attribution (last-layer [CLS] query):")
    if attributions is None:
        print("  unavailable for this model")
        return
    if not attributions:
        print("  no valid non-special tokens found")
        return

    for rank, item in enumerate(attributions, start=1):
        token = str(item["token"]).replace("\n", " ").strip()
        phrase = str(item["phrase"]).replace("\n", " ").strip()
        span_text = ""
        if "token_char_start" in item and "phrase_char_start" in item:
            span_text = (
                f" token_span=({item['token_char_start']},{item['token_char_end']})"
                f" phrase_span=({item['phrase_char_start']},{item['phrase_char_end']})"
            )
        print(
            f"  {rank}. token='{token}' idx={item['token_index']} "
            f"score={item['token_score']:.4f} "
            f"phrase='{phrase}' phrase_len={item['phrase_len']} "
            f"phrase_avg_score={item['phrase_score']:.4f} "
            f"phrase_total_score={item['phrase_total_score']:.4f}"
            f"{span_text}"
        )


def _select_outstanding_tokens(
    attributions: list[dict[str, Any]],
    min_token_score: float,
    median_ratio: float,
    max_ratio: float,
) -> list[dict[str, Any]]:
    token_map: dict[tuple[int, int], dict[str, Any]] = {}
    for item in attributions:
        if "token_char_start" not in item or "token_char_end" not in item:
            continue
        start = int(item["token_char_start"])
        end = int(item["token_char_end"])
        if end <= start:
            continue
        score = float(item["token_score"])
        key = (start, end)
        prev = token_map.get(key)
        if prev is None or score > float(prev["score"]):
            token_map[key] = {
                "start": start,
                "end": end,
                "token": str(item["token"]),
                "score": score,
            }

    candidates = sorted(token_map.values(), key=lambda x: float(x["score"]), reverse=True)
    if not candidates:
        return []
    # Keep all sufficiently strong candidates; iterative removal will use
    # probability-drop validation to choose among them.
    return [c for c in candidates if float(c["score"]) >= min_token_score]


def _mask_text_by_spans(text: str, spans: list[tuple[int, int]], mask_token: str) -> str:
    if not spans:
        return text
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)

    masked = text
    for start, end in reversed([(s, e) for s, e in merged]):
        masked = masked[:start] + f" {mask_token} " + masked[end:]
    return " ".join(masked.split())


def _rank_tokens_by_prob_drop(
    *,
    args: argparse.Namespace,
    standout: list[dict[str, Any]],
    current_text: str,
    current_result: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
    mask_token: str,
) -> list[dict[str, Any]]:
    target_idx = int(current_result["pred"])
    base_prob = float(current_result["probs"][target_idx].item())
    eval_candidates = standout[: max(args.attribution_iter_eval_top_n, 0)]

    scored: list[dict[str, Any]] = []
    for item in eval_candidates:
        masked_text = _mask_text_by_spans(
            current_text,
            [(int(item["start"]), int(item["end"]))],
            mask_token=mask_token,
        )
        if masked_text == current_text:
            continue
        masked_result = _run_single_prediction(
            args=args,
            text=masked_text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            with_attribution=False,
        )
        masked_prob = float(masked_result["probs"][target_idx].item())
        prob_drop = base_prob - masked_prob
        scored_item = dict(item)
        scored_item["prob_drop"] = prob_drop
        scored_item["abs_prob_change"] = abs(prob_drop)
        scored.append(scored_item)

    scored.sort(key=lambda x: (float(x["prob_drop"]), float(x["score"])), reverse=True)
    return scored


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_dir)
    model.to(device)
    model.eval()

    result = _run_single_prediction(
        args=args,
        text=args.text,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    print("label:", ID_TO_LABEL[int(result["pred"])])
    _print_probabilities(result["probs"])
    if args.show_attribution:
        _print_attributions(result["attributions"])

    if not args.attribution_iterative_erasure:
        return
    if not args.show_attribution:
        print("\niterative_erasure: requires --show_attribution")
        return
    if result["attributions"] is None:
        print("\niterative_erasure: unavailable (model did not return attentions)")
        return

    mask_token = args.attribution_iter_mask_token
    if mask_token is None:
        mask_token = tokenizer.mask_token or "[MASK]"
    current_text = args.text

    for round_idx in range(1, args.attribution_iter_max_rounds + 1):
        current_attributions = result["attributions"] or []
        standout = _select_outstanding_tokens(
            attributions=current_attributions,
            min_token_score=args.attribution_iter_min_token_score,
            median_ratio=args.attribution_iter_median_ratio,
            max_ratio=args.attribution_iter_max_ratio,
        )
        if not standout:
            print(f"\niterative_erasure: stop at round {round_idx - 1} (no standout tokens)")
            return

        ranked_by_drop = _rank_tokens_by_prob_drop(
            args=args,
            standout=standout,
            current_text=current_text,
            current_result=result,
            tokenizer=tokenizer,
            model=model,
            device=device,
            mask_token=mask_token,
        )
        chosen = [
            item
            for item in ranked_by_drop
            if float(item["prob_drop"]) >= args.attribution_iter_min_prob_drop
        ][: max(args.attribution_iter_remove_top_n, 1)]
        selection_mode = "positive_drop"
        if not chosen:
            print(
                f"\niterative_erasure: stop at round {round_idx - 1} "
                "(no standout token causes enough probability drop)"
            )
            return

        spans = [(int(item["start"]), int(item["end"])) for item in chosen]
        tokens = [
            (
                f"{str(item['token']).strip()}(drop={float(item['prob_drop']):.4f},"
                f"abs={float(item['abs_prob_change']):.4f})"
            )
            for item in chosen
        ]
        next_text = _mask_text_by_spans(current_text, spans, mask_token=mask_token)
        if next_text == current_text:
            print(f"\niterative_erasure: stop at round {round_idx - 1} (no text change)")
            return

        result = _run_single_prediction(
            args=args,
            text=next_text,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        print(f"\niterative_erasure round {round_idx}:")
        print(f"  selection_mode: {selection_mode}")
        print(f"  masked_tokens: {', '.join(tokens)}")
        print(f"  text: {next_text}")
        print("  label:", ID_TO_LABEL[int(result["pred"])])
        print("  probabilities:")
        for i, prob in enumerate(result["probs"].tolist()):
            print(f"    {ID_TO_LABEL[i]}: {prob:.4f}")
        _print_attributions(result["attributions"])
        current_text = next_text


if __name__ == "__main__":
    main()
