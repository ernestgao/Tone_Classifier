from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "why",
    "with",
    "you",
    "your",
}


def _find_cls_index(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> int:
    cls_token_id = tokenizer.cls_token_id
    if cls_token_id is not None:
        cls_positions = torch.nonzero(input_ids == cls_token_id, as_tuple=False)
        if len(cls_positions) > 0:
            return int(cls_positions[0].item())
    return 0


def _decode_span(
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    start_idx: int,
    end_idx: int,
) -> str:
    token_ids = input_ids[start_idx : end_idx + 1].tolist()
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return text.strip()


def _extract_span_text_from_offsets(
    text: str,
    offsets: list[tuple[int, int]] | None,
    start_idx: int,
    end_idx: int,
) -> str:
    if offsets is None:
        return ""
    start_char, _ = offsets[start_idx]
    _, end_char = offsets[end_idx]
    if end_char <= start_char:
        return ""
    return text[start_char:end_char].strip()


def _build_special_mask(
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    special_mask = tokenizer.get_special_tokens_mask(
        input_ids.tolist(),
        already_has_special_tokens=True,
    )
    return torch.tensor(special_mask, dtype=torch.bool, device=input_ids.device)


def _expand_phrase_indices(
    *,
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    start_idx: int,
    max_phrase_tokens: int,
    drop_vs_initial_threshold: float,
    small_drop_no_change_threshold: float,
    drop_vs_prev_threshold: float,
) -> list[tuple[int, int]]:
    end_idx = start_idx
    spans: list[tuple[int, int]] = [(start_idx, end_idx)]
    initial_score = float(scores[start_idx].item())
    prev_score = initial_score

    while len(spans) < max_phrase_tokens:
        next_idx = end_idx + 1
        if next_idx >= scores.shape[0] or not bool(candidate_mask[next_idx].item()):
            break

        new_score = float(scores[start_idx : next_idx + 1].mean().item())
        if _should_stop_expansion(
            previous_score=prev_score,
            new_score=new_score,
            initial_score=initial_score,
            drop_vs_initial_threshold=drop_vs_initial_threshold,
            small_drop_no_change_threshold=small_drop_no_change_threshold,
            drop_vs_prev_threshold=drop_vs_prev_threshold,
        ):
            break

        end_idx = next_idx
        prev_score = new_score
        spans.append((start_idx, end_idx))

    return spans


def _should_stop_expansion(
    *,
    previous_score: float,
    new_score: float,
    initial_score: float,
    drop_vs_initial_threshold: float,
    small_drop_no_change_threshold: float,
    drop_vs_prev_threshold: float,
) -> bool:
    eps = 1e-12
    if previous_score <= eps:
        return False

    ratio_prev = new_score / previous_score
    ratio_initial = new_score / initial_score if initial_score > eps else ratio_prev

    if new_score < previous_score:
        if drop_vs_prev_threshold > 0 and ratio_prev < drop_vs_prev_threshold:
            return True
        if drop_vs_initial_threshold > 0 and ratio_initial < drop_vs_initial_threshold:
            return True
        if small_drop_no_change_threshold > 0 and ratio_prev > small_drop_no_change_threshold:
            return True
        return False

    if new_score > previous_score:
        return False

    return True


def _normalized_text_key(text: str) -> str:
    return "".join(ch for ch in text.lower().strip() if ch.isalnum() or ch.isspace())


def _is_content_phrase(
    phrase: str,
    phrase_len: int,
    min_alpha_chars: int,
) -> bool:
    normalized = phrase.lower().strip()
    if not normalized:
        return False
    alpha_only = "".join(ch for ch in normalized if ch.isalpha())
    if len(alpha_only) < min_alpha_chars:
        return False
    if phrase_len == 1 and normalized in _STOPWORDS:
        return False
    return True


def extract_cls_attention_attribution(
    *,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    last_layer_attention: torch.Tensor,
    top_k: int,
    neighbor_threshold: float | None = None,
    max_phrase_tokens: int,
    max_overlap_ratio: float = 0.8,
    drop_vs_initial_threshold: float = 0.75,
    small_drop_no_change_threshold: float = 0.90,
    drop_vs_prev_threshold: float = 0.80,
    content_words_only: bool = False,
    dedup_by_text: bool = True,
    min_alpha_chars: int = 3,
    offsets: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """
    Extract token and phrase attributions from last-layer [CLS]-query attention.

    Args:
        input_ids: shape [seq_len]
        attention_mask: shape [seq_len]
        last_layer_attention: shape [num_heads, seq_len, seq_len]
    """
    if top_k <= 0:
        return []

    cls_index = _find_cls_index(input_ids, tokenizer)
    attention_mask = attention_mask.bool()
    special_mask = _build_special_mask(tokenizer, input_ids)
    candidate_mask = attention_mask & ~special_mask

    cls_attention = last_layer_attention[:, cls_index, :].mean(dim=0)
    cls_attention = torch.where(attention_mask, cls_attention, torch.zeros_like(cls_attention))

    candidate_total = cls_attention[candidate_mask].sum()
    if float(candidate_total.item()) > 0:
        normalized_scores = cls_attention / candidate_total
    else:
        normalized_scores = cls_attention

    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
    if candidate_indices.numel() == 0:
        return []

    candidate_attributions: list[dict[str, Any]] = []
    for token_idx_tensor in candidate_indices:
        token_idx = int(token_idx_tensor.item())
        span_chain = _expand_phrase_indices(
            scores=normalized_scores,
            candidate_mask=candidate_mask,
            start_idx=token_idx,
            max_phrase_tokens=max_phrase_tokens,
            drop_vs_initial_threshold=drop_vs_initial_threshold,
            small_drop_no_change_threshold=small_drop_no_change_threshold,
            drop_vs_prev_threshold=drop_vs_prev_threshold,
        )

        token_score = float(normalized_scores[token_idx].item())
        token_text = _extract_span_text_from_offsets(text, offsets, token_idx, token_idx)
        if not token_text:
            token_text = _decode_span(tokenizer, input_ids, token_idx, token_idx)

        for start_idx, end_idx in span_chain:
            phrase_text = _extract_span_text_from_offsets(text, offsets, start_idx, end_idx)
            if not phrase_text:
                phrase_text = _decode_span(tokenizer, input_ids, start_idx, end_idx)

            span_scores = normalized_scores[start_idx : end_idx + 1]
            phrase_score = float(span_scores.mean().item())
            phrase_total_score = float(span_scores.sum().item())
            phrase_len = end_idx - start_idx + 1

            attribution: dict[str, Any] = {
                "token_index": token_idx,
                "token": token_text,
                "token_score": token_score,
                "phrase_token_start": start_idx,
                "phrase_token_end": end_idx,
                "phrase": phrase_text,
                "phrase_score": phrase_score,
                "phrase_total_score": phrase_total_score,
                "phrase_len": phrase_len,
            }

            if offsets is not None:
                token_start, token_end = offsets[token_idx]
                phrase_start, _ = offsets[start_idx]
                _, phrase_end = offsets[end_idx]
                attribution["token_char_start"] = token_start
                attribution["token_char_end"] = token_end
                attribution["phrase_char_start"] = phrase_start
                attribution["phrase_char_end"] = phrase_end

            candidate_attributions.append(attribution)

    if not candidate_attributions:
        return []

    deduped_by_span: dict[tuple[int, int], dict[str, Any]] = {}
    for item in candidate_attributions:
        span_key = (int(item["phrase_token_start"]), int(item["phrase_token_end"]))
        prev = deduped_by_span.get(span_key)
        if prev is None or float(item["phrase_score"]) > float(prev["phrase_score"]):
            deduped_by_span[span_key] = item

    ranked = sorted(
        deduped_by_span.values(),
        key=lambda x: (float(x["phrase_score"]), float(x["token_score"])),
        reverse=True,
    )

    if content_words_only:
        ranked = [
            item
            for item in ranked
            if _is_content_phrase(
                phrase=str(item["phrase"]),
                phrase_len=int(item["phrase_len"]),
                min_alpha_chars=min_alpha_chars,
            )
        ]

    if dedup_by_text:
        deduped_text: dict[str, dict[str, Any]] = {}
        for item in ranked:
            key = _normalized_text_key(str(item["phrase"]))
            if not key:
                continue
            prev = deduped_text.get(key)
            if prev is None or float(item["phrase_score"]) > float(prev["phrase_score"]):
                deduped_text[key] = item
        ranked = sorted(
            deduped_text.values(),
            key=lambda x: (float(x["phrase_score"]), float(x["token_score"])),
            reverse=True,
        )

    if max_overlap_ratio <= 0:
        return ranked[:top_k]

    selected: list[dict[str, Any]] = []
    for item in ranked:
        start_a = int(item["phrase_token_start"])
        end_a = int(item["phrase_token_end"])
        len_a = end_a - start_a + 1

        too_similar = False
        for chosen in selected:
            start_b = int(chosen["phrase_token_start"])
            end_b = int(chosen["phrase_token_end"])
            len_b = end_b - start_b + 1

            inter = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
            if inter == 0:
                continue
            overlap_ratio = inter / float(max(len_a, len_b))
            if overlap_ratio >= max_overlap_ratio:
                too_similar = True
                break

        if too_similar:
            continue

        selected.append(item)
        if len(selected) >= top_k:
            break

    return selected


def select_top_spans_for_masking(
    *,
    attributions: list[dict[str, Any]],
    top_k: int,
    use_phrase_spans: bool = False,
    max_overlap_ratio: float = 0.8,
) -> list[dict[str, Any]]:
    """
    Select top-k character spans from attribution outputs for masking.

    Returns items with: start, end, text, score.
    """
    if top_k <= 0 or not attributions:
        return []

    start_key = "phrase_char_start" if use_phrase_spans else "token_char_start"
    end_key = "phrase_char_end" if use_phrase_spans else "token_char_end"
    text_key = "phrase" if use_phrase_spans else "token"
    score_key = "phrase_score" if use_phrase_spans else "token_score"

    best_by_span: dict[tuple[int, int], dict[str, Any]] = {}
    for item in attributions:
        if start_key not in item or end_key not in item:
            continue
        start = int(item[start_key])
        end = int(item[end_key])
        if end <= start:
            continue
        span_key = (start, end)
        score = float(item.get(score_key, 0.0))
        prev = best_by_span.get(span_key)
        if prev is None or score > float(prev["score"]):
            best_by_span[span_key] = {
                "start": start,
                "end": end,
                "text": str(item.get(text_key, "")).strip(),
                "score": score,
            }

    ranked = sorted(best_by_span.values(), key=lambda x: float(x["score"]), reverse=True)
    if max_overlap_ratio <= 0:
        return ranked[:top_k]

    selected: list[dict[str, Any]] = []
    for item in ranked:
        start_a = int(item["start"])
        end_a = int(item["end"])
        len_a = end_a - start_a
        if len_a <= 0:
            continue

        too_similar = False
        for chosen in selected:
            start_b = int(chosen["start"])
            end_b = int(chosen["end"])
            len_b = end_b - start_b
            if len_b <= 0:
                continue

            inter = max(0, min(end_a, end_b) - max(start_a, start_b))
            if inter == 0:
                continue
            overlap_ratio = inter / float(max(len_a, len_b))
            if overlap_ratio >= max_overlap_ratio:
                too_similar = True
                break
        if too_similar:
            continue

        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected


def merge_character_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(s, e) for s, e in merged]


def mask_text_by_character_spans(
    text: str,
    spans: list[tuple[int, int]],
    mask_token: str,
) -> str:
    if not spans:
        return text
    merged = merge_character_spans(spans)
    if not merged:
        return text

    masked = text
    for start, end in reversed(merged):
        masked = masked[:start] + f" {mask_token} " + masked[end:]
    return " ".join(masked.split())
