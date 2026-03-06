from __future__ import annotations

import argparse
import re
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer

from tone_classifier.attribution_ranking import (
    extract_cls_attention_attribution,
    mask_text_by_character_spans,
    select_top_spans_for_masking,
)


ID_TO_LABEL = {
    0: "impolite",
    1: "neutral",
    2: "polite",
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


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
    p.add_argument(
        "--attribution_mask_top_k",
        type=int,
        default=0,
        help="Mask top-k attributed spans in the original prompt.",
    )
    p.add_argument(
        "--attribution_mask_use_phrases",
        action="store_true",
        help="Mask phrase spans instead of single-token spans.",
    )
    p.add_argument(
        "--attribution_mask_token",
        type=str,
        default=None,
        help="Mask token for top-k masking. Defaults to tokenizer mask token or [MASK].",
    )
    p.add_argument(
        "--fill_masks_with_mlm",
        action="store_true",
        help="Fill generated mask tokens using a (fine-tuned) masked language model.",
    )
    p.add_argument(
        "--mlm_model_dir",
        type=str,
        default=None,
        help="Path/name of fine-tuned masked language model for mask filling.",
    )
    p.add_argument(
        "--mlm_candidate_top_k",
        type=int,
        default=30,
        help="Candidate pool per mask position before reranking.",
    )
    p.add_argument(
        "--mlm_rerank_top_k",
        type=int,
        default=6,
        help="How many MLM candidates to rerank with classifier probability.",
    )
    p.add_argument(
        "--mlm_target_label",
        type=str,
        choices=["impolite", "neutral", "polite"],
        default="neutral",
        help="Target label for optional classifier-based candidate reranking.",
    )
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


def _normalize_mlm_token(token: str) -> str:
    t = token.strip()
    if t.startswith("##"):
        return ""
    if t.startswith("Ġ") or t.startswith("▁"):
        t = t[1:]
    t = t.strip()
    if not t:
        return ""
    if not re.match(r"^[A-Za-z][A-Za-z'\-]*$", t):
        return ""
    return t


def _replace_first_mask(text: str, mask_token: str, replacement: str) -> str:
    pattern = re.escape(mask_token)
    return re.sub(pattern, replacement, text, count=1)


def _build_topk_masked_prompt(
    *,
    text: str,
    attributions: list[dict[str, Any]] | None,
    top_k: int,
    use_phrase_spans: bool,
    mask_token: str,
    overlap_ratio: float,
) -> tuple[str, list[dict[str, Any]]]:
    if not attributions or top_k <= 0:
        return text, []
    selected = select_top_spans_for_masking(
        attributions=attributions,
        top_k=top_k,
        use_phrase_spans=use_phrase_spans,
        max_overlap_ratio=overlap_ratio,
    )
    spans = [(int(x["start"]), int(x["end"])) for x in selected]
    masked_text = mask_text_by_character_spans(text, spans, mask_token=mask_token)
    return masked_text, selected


def _fill_masks_with_mlm(
    *,
    args: argparse.Namespace,
    masked_text: str,
    mask_token: str,
    mlm_tokenizer: Any,
    mlm_model: Any,
    classifier_tokenizer: Any,
    classifier_model: Any,
    classifier_device: str,
) -> tuple[str, list[dict[str, Any]]]:
    mask_token_id = mlm_tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("MLM tokenizer has no mask token id.")

    if mask_token not in masked_text:
        return masked_text, []

    mlm_device = next(mlm_model.parameters()).device
    target_label_id = LABEL_TO_ID[args.mlm_target_label]
    current_text = masked_text
    fill_steps: list[dict[str, Any]] = []
    max_loops = current_text.count(mask_token)

    for step_idx in range(max_loops):
        inputs = mlm_tokenizer(
            current_text,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(mlm_device) for k, v in inputs.items()}
        mask_positions = torch.nonzero(
            inputs["input_ids"][0] == mask_token_id,
            as_tuple=False,
        ).squeeze(-1)
        if mask_positions.numel() == 0:
            break

        pos = int(mask_positions[0].item())
        with torch.no_grad():
            logits = mlm_model(**inputs).logits[0, pos]

        candidate_pool = max(args.mlm_candidate_top_k, args.mlm_rerank_top_k, 1)
        top_ids = torch.topk(logits, k=candidate_pool).indices.tolist()
        candidates: list[str] = []
        for token_id in top_ids:
            raw_token = mlm_tokenizer.convert_ids_to_tokens(int(token_id))
            cand = _normalize_mlm_token(str(raw_token))
            if not cand:
                continue
            if cand.lower() not in {x.lower() for x in candidates}:
                candidates.append(cand)
            if len(candidates) >= candidate_pool:
                break
        if not candidates:
            break

        rerank_pool = candidates[: max(args.mlm_rerank_top_k, 1)]
        best = rerank_pool[0]
        best_score = float("-inf")
        for cand in rerank_pool:
            trial_text = _replace_first_mask(current_text, mask_token, cand)
            trial_result = _run_single_prediction(
                args=args,
                text=trial_text,
                tokenizer=classifier_tokenizer,
                model=classifier_model,
                device=classifier_device,
                with_attribution=False,
            )
            score = float(trial_result["probs"][target_label_id].item())
            if score > best_score:
                best_score = score
                best = cand

        current_text = _replace_first_mask(current_text, mask_token, best)
        fill_steps.append(
            {
                "step": step_idx + 1,
                "chosen": best,
                "target_label_prob": best_score,
                "candidates": rerank_pool,
            }
        )

    return current_text, fill_steps


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
        masked_text = mask_text_by_character_spans(
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

    mlm_tokenizer = None
    mlm_model = None
    if args.fill_masks_with_mlm:
        if not args.mlm_model_dir:
            raise ValueError("--fill_masks_with_mlm requires --mlm_model_dir")
        mlm_tokenizer = AutoTokenizer.from_pretrained(args.mlm_model_dir)
        mlm_model = AutoModelForMaskedLM.from_pretrained(args.mlm_model_dir)
        mlm_model.to(device)
        mlm_model.eval()

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

    if args.attribution_mask_top_k > 0:
        if not args.show_attribution:
            print("\nmask_top_k: requires --show_attribution")
        elif result["attributions"] is None:
            print("\nmask_top_k: unavailable (model did not return attentions)")
        else:
            mask_token = args.attribution_mask_token
            if mask_token is None:
                mask_token = tokenizer.mask_token or "[MASK]"
            if mlm_tokenizer is not None and mlm_tokenizer.mask_token:
                mask_token = mlm_tokenizer.mask_token

            masked_text, selected_spans = _build_topk_masked_prompt(
                text=args.text,
                attributions=result["attributions"],
                top_k=args.attribution_mask_top_k,
                use_phrase_spans=args.attribution_mask_use_phrases,
                mask_token=mask_token,
                overlap_ratio=args.attribution_max_overlap_ratio,
            )
            print("\nmasked_prompt_from_top_k:")
            print(f"  mask_token: {mask_token}")
            if not selected_spans:
                print("  selected_spans: none")
            else:
                print("  selected_spans:")
                for i, span in enumerate(selected_spans, start=1):
                    print(
                        "    "
                        f"{i}. text='{span['text']}' "
                        f"char_span=({int(span['start'])},{int(span['end'])}) "
                        f"score={float(span['score']):.4f}"
                    )
            print(f"  masked_text: {masked_text}")

            if args.fill_masks_with_mlm:
                filled_text, fill_steps = _fill_masks_with_mlm(
                    args=args,
                    masked_text=masked_text,
                    mask_token=mask_token,
                    mlm_tokenizer=mlm_tokenizer,
                    mlm_model=mlm_model,
                    classifier_tokenizer=tokenizer,
                    classifier_model=model,
                    classifier_device=device,
                )
                print("\nmlm_fill_steps:")
                if not fill_steps:
                    print("  none (no mask tokens filled)")
                else:
                    for step in fill_steps:
                        print(
                            "  "
                            f"step={int(step['step'])} chosen='{step['chosen']}' "
                            f"target_prob={float(step['target_label_prob']):.4f} "
                            f"candidates={step['candidates']}"
                        )
                print(f"  filled_text: {filled_text}")
                filled_result = _run_single_prediction(
                    args=args,
                    text=filled_text,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    with_attribution=False,
                )
                print("  filled_text_label:", ID_TO_LABEL[int(filled_result["pred"])])
                print("  filled_text_probabilities:")
                for i, prob in enumerate(filled_result["probs"].tolist()):
                    print(f"    {ID_TO_LABEL[i]}: {prob:.4f}")

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
        next_text = mask_text_by_character_spans(current_text, spans, mask_token=mask_token)
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
