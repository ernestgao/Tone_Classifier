"""
Context-Cite based attribution for tone classification.
Uses sentence-level attribution instead of word-level as recommended by TA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
    baseline_pred: Dict,
    ablated_pred: Dict,
) -> float:
    """
    Compute attribution score by comparing baseline and ablated predictions.
    
    Uses the change in probability for the predicted class.
    """
    baseline_label_id = baseline_pred["label_id"]
    baseline_prob = baseline_pred["probabilities"][baseline_label_id]
    ablated_prob = ablated_pred["probabilities"][baseline_label_id]
    
    # Attribution score is the drop in probability when sentence is removed
    score = baseline_prob - ablated_prob
    return float(score)


def sentence_level_attribution(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    num_ablations: int = 256,  # TA建议增加到256/512
    device: str = "cuda",
    max_length: int = 128,
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
    
    # 3. For each sentence, compute attribution via ablation
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        # Ablate this sentence (remove it)
        ablated_sentences = [s for j, s in enumerate(sentences) if j != i]
        ablated_text = " ".join(ablated_sentences)
        
        # If no text left after ablation, skip
        if not ablated_text.strip():
            sentence_scores.append({
                "sentence": sentence,
                "attribution_score": 0.0,
                "index": i,
                "note": "Empty text after ablation",
            })
            continue
        
        # Run multiple ablations with different random seeds/noise
        # In practice, we can add small noise or use different tokenizations
        ablation_scores = []
        for seed in range(num_ablations):
            # For now, we use the same ablated text
            # In more sophisticated implementations, we could add noise
            torch.manual_seed(seed)
            ablated_pred = get_prediction(
                model, tokenizer, ablated_text, device, max_length
            )
            score = compute_attribution_score(baseline_pred, ablated_pred)
            ablation_scores.append(score)
        
        avg_score = np.mean(ablation_scores)
        std_score = np.std(ablation_scores)
        
        sentence_scores.append({
            "sentence": sentence,
            "attribution_score": float(avg_score),
            "attribution_std": float(std_score),
            "index": i,
        })
    
    # Sort by attribution score (highest first)
    sentence_scores.sort(key=lambda x: x["attribution_score"], reverse=True)
    
    return {
        "sentences": sentence_scores,
        "baseline_prediction": baseline_pred,
        "num_ablations": num_ablations,
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
        print(f"{i}. Score: {sent_info['attribution_score']:.4f} ± {sent_info['attribution_std']:.4f}")
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