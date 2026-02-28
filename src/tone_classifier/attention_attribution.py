"""
Attention-based attribution methods for tone classification.
Extracts and analyzes attention weights from transformer models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tone_classifier.data import ID_TO_LABEL


def extract_attention_weights(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cuda",
    max_length: int = 128,
    layer_idx: int = -1,  # -1 means last layer
) -> Dict:
    """
    Extract attention weights from the model.
    
    Args:
        model: Fine-tuned classification model
        tokenizer: Tokenizer
        text: Input text
        device: Device to run on
        max_length: Maximum sequence length
        layer_idx: Which layer to extract from (-1 for last layer)
    
    Returns:
        Dictionary with attention weights and token information
    """
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Forward pass with output_attentions=True
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
    
    # Extract attention weights
    # outputs.attentions is a tuple of tensors, one per layer
    # Each tensor shape: (batch_size, num_heads, seq_len, seq_len)
    all_attentions = outputs.attentions
    
    # Get the specified layer (or last layer)
    if layer_idx < 0:
        layer_idx = len(all_attentions) + layer_idx
    
    attention_weights = all_attentions[layer_idx]  # Shape: (batch, heads, seq, seq)
    attention_weights = attention_weights.squeeze(0)  # Remove batch dim: (heads, seq, seq)
    
    # Average across attention heads
    avg_attention = attention_weights.mean(dim=0)  # Shape: (seq, seq)
    
    # Get attention to [CLS] token (usually index 0) or to the classification head
    # For classification, we're interested in how tokens attend to each other
    # and how they contribute to the final prediction
    
    # Method 1: Attention from [CLS] token to all tokens
    cls_attention = avg_attention[0, :].cpu().numpy()  # Attention from CLS to all tokens
    
    # Method 2: Average attention received by each token (column sum)
    token_importance = avg_attention.sum(dim=0).cpu().numpy()  # How much each token is attended to
    
    # Method 3: Attention to the last token (often used for classification)
    if avg_attention.shape[0] > 1:
        last_token_attention = avg_attention[-1, :].cpu().numpy()
    else:
        last_token_attention = cls_attention
    
    return {
        "tokens": tokens,
        "input_ids": input_ids[0].cpu().numpy().tolist(),
        "attention_weights": avg_attention.cpu().numpy().tolist(),
        "cls_attention": cls_attention.tolist(),
        "token_importance": token_importance.tolist(),
        "last_token_attention": last_token_attention.tolist(),
        "layer_idx": layer_idx,
        "num_layers": len(all_attentions),
        "num_heads": attention_weights.shape[0],
    }


def aggregate_attention_across_layers(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cuda",
    max_length: int = 128,
    aggregation_method: str = "mean",  # "mean", "last", "weighted"
) -> Dict:
    """
    Aggregate attention weights across all layers.
    
    Args:
        aggregation_method: How to aggregate across layers
            - "mean": Average all layers
            - "last": Use only last layer
            - "weighted": Weight later layers more heavily
    """
    model.eval()
    
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
    
    all_attentions = outputs.attentions
    num_layers = len(all_attentions)
    
    # Aggregate across layers
    if aggregation_method == "mean":
        # Average all layers
        aggregated = torch.stack([att.squeeze(0).mean(dim=0) for att in all_attentions])
        aggregated = aggregated.mean(dim=0)  # Average across layers
    
    elif aggregation_method == "last":
        # Use only last layer
        aggregated = all_attentions[-1].squeeze(0).mean(dim=0)
    
    elif aggregation_method == "weighted":
        # Weight later layers more heavily
        weights = torch.linspace(0.5, 1.0, num_layers).to(device)
        weights = weights / weights.sum()
        
        layer_attentions = torch.stack([att.squeeze(0).mean(dim=0) for att in all_attentions])
        aggregated = (layer_attentions * weights.view(-1, 1, 1)).sum(dim=0)
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    # Get token importance scores
    cls_attention = aggregated[0, :].cpu().numpy()
    token_importance = aggregated.sum(dim=0).cpu().numpy()
    
    return {
        "tokens": tokens,
        "cls_attention": cls_attention.tolist(),
        "token_importance": token_importance.tolist(),
        "aggregation_method": aggregation_method,
        "num_layers": num_layers,
    }


def attention_to_sentence_attribution(
    attention_results: Dict,
    text: str,
) -> Dict:
    """
    Convert token-level attention to sentence-level attribution.
    """
    from tone_classifier.attribution import split_into_sentences
    
    sentences = split_into_sentences(text)
    tokens = attention_results["tokens"]
    token_importance = np.array(attention_results["token_importance"])
    
    # Map tokens to sentences (simplified - in practice need better tokenization alignment)
    sentence_scores = []
    
    # Simple heuristic: divide tokens roughly equally among sentences
    tokens_per_sentence = len(tokens) // max(len(sentences), 1)
    
    for i, sentence in enumerate(sentences):
        start_idx = i * tokens_per_sentence
        end_idx = (i + 1) * tokens_per_sentence if i < len(sentences) - 1 else len(tokens)
        
        # Sum attention scores for tokens in this sentence
        sentence_score = token_importance[start_idx:end_idx].sum()
        
        sentence_scores.append({
            "sentence": sentence,
            "attribution_score": float(sentence_score),
            "index": i,
            "token_range": (start_idx, end_idx),
        })
    
    # Sort by score
    sentence_scores.sort(key=lambda x: x["attribution_score"], reverse=True)
    
    return {
        "sentences": sentence_scores,
        "method": "attention_based",
        "total_sentences": len(sentences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract attention-based attribution from model"
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
        "--layer_idx",
        type=int,
        default=-1,
        help="Which layer to extract (-1 for last layer)",
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="mean",
        choices=["mean", "last", "weighted"],
        help="How to aggregate across layers",
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
    parser.add_argument(
        "--sentence_level",
        action="store_true",
        help="Convert to sentence-level attribution",
    )
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.hf_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_dir)
    model.to(args.device)
    model.eval()
    
    # Extract attention
    print(f"Extracting attention weights...")
    if args.aggregation_method != "last" or args.layer_idx == -1:
        results = aggregate_attention_across_layers(
            model=model,
            tokenizer=tokenizer,
            text=args.text,
            device=args.device,
            max_length=args.max_length,
            aggregation_method=args.aggregation_method,
        )
    else:
        results = extract_attention_weights(
            model=model,
            tokenizer=tokenizer,
            text=args.text,
            device=args.device,
            max_length=args.max_length,
            layer_idx=args.layer_idx,
        )
    
    # Convert to sentence level if requested
    if args.sentence_level:
        sentence_results = attention_to_sentence_attribution(results, args.text)
        results.update(sentence_results)
    
    # Print results
    print("\n" + "=" * 80)
    print("ATTENTION-BASED ATTRIBUTION RESULTS")
    print("=" * 80)
    
    if args.sentence_level:
        print(f"\nSentence Attribution Scores (sorted by importance):")
        for i, sent_info in enumerate(results['sentences'], 1):
            print(f"{i}. Score: {sent_info['attribution_score']:.4f}")
            print(f"   Sentence: {sent_info['sentence'][:100]}...")
            print()
    else:
        print(f"\nTop 10 Most Important Tokens:")
        token_scores = list(zip(results['tokens'], results['token_importance']))
        token_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (token, score) in enumerate(token_scores[:10], 1):
            print(f"{i}. {token}: {score:.4f}")
    
    # Save to file if specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
