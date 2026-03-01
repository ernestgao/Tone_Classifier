"""
Context-Cite-inspired attribution for tone classification.
Uses sentence-level attribution instead of word-level as recommended by TA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tone_classifier.data import ID_TO_LABEL


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    Simple implementation - can be improved with nltk if needed.
    """
    import re
    
    # Simple sentence splitting on punctuation
    # Keep punctuation with the sentence
    sentences = re.split(r'([.!?]+)', text)
    
    # Recombine sentences with their punctuation
    result = []
    current = ""
    for part in sentences:
        if re.match(r'[.!?]+', part):
            current += part
            if current.strip():
                result.append(current.strip())
            current = ""
        else:
            current += part
    
    # Add remaining text if any
    if current.strip():
        result.append(current.strip())
    
    # If no sentences found, return the whole text as one sentence
    if not result:
        return [text]
    
    return result


def get_prediction(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cuda",
    max_length: int = 128,
) -> Dict:
    """
    Get model prediction for a given text.
    
    Returns:
        Dictionary with prediction label, probabilities, and logits
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred = int(torch.argmax(probs).item())
    
    return {
        "label": ID_TO_LABEL[pred],
        "label_id": pred,
        "probabilities": probs.cpu().numpy().tolist(),
        "logits": logits.cpu().numpy().tolist(),
    }


def compute_attribution_score(
    with_target_pred: Dict,
    without_target_pred: Dict,
    target_label_id: int,
) -> float:
    """
    Compute marginal contribution score for a sentence in a sampled context.

    Score is the probability gain for `target_label_id` when the target sentence
    is present vs absent under the same sampled context.
    """
    with_prob = with_target_pred["probabilities"][target_label_id]
    without_prob = without_target_pred["probabilities"][target_label_id]
    score = with_prob - without_prob
    return float(score)


def _sample_context_indices(
    rng: np.random.Generator,
    candidate_indices: List[int],
    keep_prob: float,
    min_context_sentences: int,
) -> List[int]:
    """
    Sample a subset of context sentence indices.
    """
    if not candidate_indices:
        return []

    sampled = [idx for idx in candidate_indices if rng.random() < keep_prob]
    min_required = max(0, min_context_sentences)

    if len(sampled) < min_required:
        remaining = [idx for idx in candidate_indices if idx not in sampled]
        if remaining:
            need = min(min_required - len(sampled), len(remaining))
            sampled.extend(rng.choice(remaining, size=need, replace=False).tolist())

    sampled.sort()
    return sampled


def _join_sentences_by_indices(sentences: List[str], indices: List[int]) -> str:
    """
    Reconstruct text by sentence indices while preserving original order.
    """
    return " ".join(sentences[idx] for idx in sorted(indices))


def sentence_level_attribution(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    num_ablations: int = 256,  # TA建议增加到256/512
    device: str = "cuda",
    max_length: int = 128,
    context_keep_prob: float = 0.8,
    min_context_sentences: int = 1,
    random_seed: int = 42,
) -> Dict:
    """
    Compute sentence-level attribution scores using context-cite approach.
    
    Args:
        model: Fine-tuned classification model
        tokenizer: Tokenizer
        text: Input text
        num_ablations: Number of ablation experiments (TA建议256或512)
        device: Device to run on
        max_length: Maximum sequence length
        context_keep_prob: Probability of keeping each non-target sentence
        min_context_sentences: Minimum context sentences in each sampled ablation
        random_seed: Random seed for reproducible context sampling
    
    Returns:
        Dictionary with sentence scores and attribution results
    """
    # 1. Split text into sentences
    sentences = split_into_sentences(text)
    
    if len(sentences) == 0:
        return {
            "sentences": [],
            "baseline_prediction": None,
            "error": "No sentences found",
        }
    
    # 2. Get baseline prediction
    baseline_pred = get_prediction(model, tokenizer, text, device, max_length)
    target_label_id = baseline_pred["label_id"]
    
    # 3. For each sentence, estimate contextual marginal contribution via
    # sampled context subsets, then also report classic leave-one-out.
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        # Leave-one-out score on full context (classic ablation)
        ablated_sentences = [s for j, s in enumerate(sentences) if j != i]
        ablated_text = " ".join(ablated_sentences)
        loo_score = 0.0
        if ablated_text.strip():
            ablated_pred = get_prediction(model, tokenizer, ablated_text, device, max_length)
            loo_score = (
                baseline_pred["probabilities"][target_label_id]
                - ablated_pred["probabilities"][target_label_id]
            )

        # Contextual sampled attribution
        context_candidates = [idx for idx in range(len(sentences)) if idx != i]
        rng = np.random.default_rng(random_seed + i)

        unique_contexts = set()
        sampled_contexts: List[List[int]] = []
        max_attempts = max(num_ablations * 10, 64)
        attempts = 0
        while len(sampled_contexts) < num_ablations and attempts < max_attempts:
            attempts += 1
            sampled = _sample_context_indices(
                rng=rng,
                candidate_indices=context_candidates,
                keep_prob=float(np.clip(context_keep_prob, 0.0, 1.0)),
                min_context_sentences=min_context_sentences,
            )
            if not sampled:
                continue
            key = tuple(sampled)
            if key in unique_contexts:
                continue
            unique_contexts.add(key)
            sampled_contexts.append(sampled)

        sampled_scores: List[float] = []
        for context in sampled_contexts:
            with_target_indices = sorted(context + [i])
            without_target_indices = context
            with_target_text = _join_sentences_by_indices(sentences, with_target_indices)
            without_target_text = _join_sentences_by_indices(sentences, without_target_indices)

            with_target_pred = get_prediction(
                model, tokenizer, with_target_text, device, max_length
            )
            without_target_pred = get_prediction(
                model, tokenizer, without_target_text, device, max_length
            )
            sampled_scores.append(
                compute_attribution_score(
                    with_target_pred=with_target_pred,
                    without_target_pred=without_target_pred,
                    target_label_id=target_label_id,
                )
            )

        if sampled_scores:
            marginal_mean = float(np.mean(sampled_scores))
            marginal_std = float(np.std(sampled_scores))
            marginal_p05 = float(np.percentile(sampled_scores, 5))
            marginal_p95 = float(np.percentile(sampled_scores, 95))
            final_score = marginal_mean
        else:
            # Single-sentence edge case or very short inputs:
            # fallback to leave-one-out so every sentence still gets a score.
            marginal_mean = float(loo_score)
            marginal_std = 0.0
            marginal_p05 = float(loo_score)
            marginal_p95 = float(loo_score)
            final_score = float(loo_score)

        sentence_scores.append({
            "sentence": sentence,
            "attribution_score": float(final_score),
            "attribution_std": float(marginal_std),  # backward-compatible alias
            "marginal_contribution_mean": float(marginal_mean),
            "marginal_contribution_std": float(marginal_std),
            "marginal_contribution_p05": float(marginal_p05),
            "marginal_contribution_p95": float(marginal_p95),
            "leave_one_out_score": float(loo_score),
            "num_effective_ablations": len(sampled_contexts),
            "index": i,
        })
    
    # Sort by attribution score (highest first)
    sentence_scores.sort(key=lambda x: x["attribution_score"], reverse=True)
    
    return {
        "sentences": sentence_scores,
        "baseline_prediction": baseline_pred,
        "num_ablations": num_ablations,
        "context_keep_prob": float(np.clip(context_keep_prob, 0.0, 1.0)),
        "min_context_sentences": int(max(0, min_context_sentences)),
        "random_seed": random_seed,
        "total_sentences": len(sentences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute sentence-level attribution using context-cite approach"
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Path to fine-tuned HuggingFace model directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text to analyze",
    )
    parser.add_argument(
        "--num_ablations",
        type=int,
        default=256,
        help="Number of ablation experiments (TA建议256或512)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file to save results",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--context_keep_prob",
        type=float,
        default=0.8,
        help="Probability of keeping each non-target sentence in sampled ablations",
    )
    parser.add_argument(
        "--min_context_sentences",
        type=int,
        default=1,
        help="Minimum context sentences kept in each sampled ablation",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible ablation sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.hf_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_dir)
    model.to(args.device)
    model.eval()
    
    # Compute attribution
    print(f"Computing attribution for text: {args.text[:100]}...")
    results = sentence_level_attribution(
        model=model,
        tokenizer=tokenizer,
        text=args.text,
        num_ablations=args.num_ablations,
        device=args.device,
        max_length=args.max_length,
        context_keep_prob=args.context_keep_prob,
        min_context_sentences=args.min_context_sentences,
        random_seed=args.random_seed,
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("ATTRIBUTION RESULTS")
    print("=" * 80)
    print(f"\nBaseline Prediction: {results['baseline_prediction']['label']}")
    print(f"Baseline Probabilities:")
    for i, prob in enumerate(results['baseline_prediction']['probabilities']):
        print(f"  {ID_TO_LABEL[i]}: {prob:.4f}")
    
    print(f"\nSentence Attribution Scores (sorted by importance):")
    print(f"Number of ablations: {results['num_ablations']}")
    print(f"Total sentences: {results['total_sentences']}\n")
    
    for i, sent_info in enumerate(results['sentences'], 1):
        print(
            f"{i}. Score: {sent_info['attribution_score']:.4f} "
            f"(std={sent_info['marginal_contribution_std']:.4f}, "
            f"LOO={sent_info['leave_one_out_score']:.4f}, "
            f"n={sent_info['num_effective_ablations']})"
        )
        print(f"   Sentence: {sent_info['sentence'][:100]}...")
        print()
    
    # Save to file if specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
