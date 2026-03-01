"""
Modal app for running attribution analysis on GPU.
Supports both gradient-based and non-gradient-based approaches.
"""
from typing import List

import modal

# Define Modal app
app = modal.App("tone-classifier-attribution")

# Define image with all dependencies
# 参考示例：使用 image.add_local_dir() 而不是 Mount
# 注意：add_local_dir必须在最后，或者使用copy=True
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.42.0",
        "accelerate>=0.31.0",
        "bitsandbytes>=0.43.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.20.0",
        "pandas>=2.0.0",
        "tiktoken",  # DeBERTa tokenizer需要
    ])
    .add_local_dir(
        ".",
        remote_path="/root/tone_classifier",
        ignore=["artifacts/", "data/", ".git/", "__pycache__/", "*.pyc", ".venv/"],
        copy=True,  # 复制文件到image中，这样可以在之后运行命令
    )
    .run_commands("cd /root/tone_classifier && pip install -e .")  # 安装tone_classifier包
)


@app.function(
    image=image,
    gpu="A100-40GB",  # Use A100 for standard models (更新为新的API格式)
    timeout=3600,
)
def run_attribution_analysis(
    model_path: str,
    text: str,
    num_ablations: int = 512,
    max_length: int = 128,
    context_keep_prob: float = 0.8,
    min_context_sentences: int = 1,
    random_seed: int = 42,
):
    """
    Run sentence-level attribution analysis on Modal GPU.
    
    Args:
        model_path: Path to model (can be HuggingFace model ID or local path)
        text: Input text to analyze
        num_ablations: Number of ablation experiments (TA建议256或512)
        max_length: Maximum sequence length
        context_keep_prob: Probability of keeping each non-target sentence
        min_context_sentences: Minimum context sentence count per ablation
        random_seed: Random seed for reproducible context sampling
    """
    import sys
    import os
    
    # 代码通过 add_local_dir 已经添加到 /root/tone_classifier
    # 添加src目录到Python路径，因为tone_classifier包在src/下
    sys.path.insert(0, "/root/tone_classifier/src")
    sys.path.insert(0, "/root/tone_classifier")
    
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from tone_classifier.attribution import sentence_level_attribution
    from tone_classifier.data import ID_TO_LABEL
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Run attribution
    print(f"Running attribution with {num_ablations} ablations...")
    results = sentence_level_attribution(
        model=model,
        tokenizer=tokenizer,
        text=text,
        num_ablations=num_ablations,
        device=device,
        max_length=max_length,
        context_keep_prob=context_keep_prob,
        min_context_sentences=min_context_sentences,
        random_seed=random_seed,
    )
    
    return results


@app.function(
    image=image,
    gpu="H100:2",  # TA建议：2*H100s/H200s for 70B models (更新为新的API格式)
    timeout=7200,
)
def run_large_model_attribution(
    model_name: str,  # e.g., "meta-llama/Llama-2-70b-hf" or other 70B models
    text: str,
    num_ablations: int = 512,
    max_length: int = 256,
    use_quantization: bool = True,  # Use 8-bit or 4-bit quantization for large models
    context_keep_prob: float = 0.8,
    min_context_sentences: int = 1,
    random_seed: int = 42,
):
    """
    Run attribution analysis with large 70B models (non-gradient-based approaches).
    Uses 2 H100 GPUs as recommended by TA.
    
    Args:
        model_name: HuggingFace model ID for large model
        text: Input text to analyze
        num_ablations: Number of ablation experiments
        max_length: Maximum sequence length
        use_quantization: Whether to use quantization to fit model in memory
        context_keep_prob: Probability of keeping each non-target sentence
        min_context_sentences: Minimum context sentence count per ablation
        random_seed: Random seed for reproducible context sampling
    """
    import sys
    # 添加src目录到Python路径，因为tone_classifier包在src/下
    sys.path.insert(0, "/root/tone_classifier/src")
    sys.path.insert(0, "/root/tone_classifier")
    
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    from tone_classifier.attribution import sentence_level_attribution
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Load model with quantization if needed
    print(f"Loading large model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if use_quantization:
        # Use 8-bit quantization to fit large models
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",  # Automatically distribute across GPUs
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
    model.eval()
    
    # Run attribution
    print(f"Running attribution with {num_ablations} ablations...")
    results = sentence_level_attribution(
        model=model,
        tokenizer=tokenizer,
        text=text,
        num_ablations=num_ablations,
        device=device,
        max_length=max_length,
        context_keep_prob=context_keep_prob,
        min_context_sentences=min_context_sentences,
        random_seed=random_seed,
    )
    
    return results


@app.function(
    image=image,
    gpu="A100-40GB",  # 更新为新的API格式
    timeout=3600,
)
def run_attention_attribution(
    model_path: str,
    text: str,
    aggregation_method: str = "mean",
    max_length: int = 128,
):
    """
    Run attention-based attribution analysis.
    """
    import sys
    # 添加src目录到Python路径，因为tone_classifier包在src/下
    sys.path.insert(0, "/root/tone_classifier/src")
    sys.path.insert(0, "/root/tone_classifier")
    
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from tone_classifier.attention_attribution import aggregate_attention_across_layers
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Run attention attribution
    print("Extracting attention weights...")
    results = aggregate_attention_across_layers(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        max_length=max_length,
        aggregation_method=aggregation_method,
    )
    
    return results


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,
)
def run_batch_attribution_analysis(
    model_path: str,
    texts: List[str],
    num_ablations: int = 512,
    max_length: int = 128,
    context_keep_prob: float = 0.8,
    min_context_sentences: int = 1,
    random_seed: int = 42,
):
    """
    Run sentence-level attribution on a batch of texts with one model load.
    """
    import sys
    import time

    # tone_classifier package lives under src/
    sys.path.insert(0, "/root/tone_classifier/src")
    sys.path.insert(0, "/root/tone_classifier")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from tone_classifier.attribution import sentence_level_attribution

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    batch_results = []
    for idx, text in enumerate(texts):
        start = time.time()
        try:
            result = sentence_level_attribution(
                model=model,
                tokenizer=tokenizer,
                text=text,
                num_ablations=num_ablations,
                device=device,
                max_length=max_length,
                context_keep_prob=context_keep_prob,
                min_context_sentences=min_context_sentences,
                random_seed=random_seed + idx,
            )
            elapsed = time.time() - start
            batch_results.append(
                {
                    "index": idx,
                    "text": text,
                    "ok": True,
                    "elapsed_seconds": elapsed,
                    "result": result,
                }
            )
        except Exception as e:
            elapsed = time.time() - start
            batch_results.append(
                {
                    "index": idx,
                    "text": text,
                    "ok": False,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                }
            )

    return {
        "model_path": model_path,
        "device": device,
        "num_inputs": len(texts),
        "num_ablations": num_ablations,
        "batch_results": batch_results,
    }


@app.function(
    image=image,
    gpu="H100:2",
    timeout=10800,
)
def run_large_model_batch_attribution(
    model_name: str,
    texts: List[str],
    num_ablations: int = 512,
    max_length: int = 256,
    use_quantization: bool = True,
    context_keep_prob: float = 0.8,
    min_context_sentences: int = 1,
    random_seed: int = 42,
):
    """
    Run sentence-level attribution on a batch with a larger model.
    """
    import sys
    import time

    sys.path.insert(0, "/root/tone_classifier/src")
    sys.path.insert(0, "/root/tone_classifier")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    from tone_classifier.attribution import sentence_level_attribution

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Loading large model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if use_quantization:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    model.eval()

    batch_results = []
    for idx, text in enumerate(texts):
        start = time.time()
        try:
            result = sentence_level_attribution(
                model=model,
                tokenizer=tokenizer,
                text=text,
                num_ablations=num_ablations,
                device=device,
                max_length=max_length,
                context_keep_prob=context_keep_prob,
                min_context_sentences=min_context_sentences,
                random_seed=random_seed + idx,
            )
            elapsed = time.time() - start
            batch_results.append(
                {
                    "index": idx,
                    "text": text,
                    "ok": True,
                    "elapsed_seconds": elapsed,
                    "result": result,
                }
            )
        except Exception as e:
            elapsed = time.time() - start
            batch_results.append(
                {
                    "index": idx,
                    "text": text,
                    "ok": False,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                }
            )

    return {
        "model_name": model_name,
        "device": device,
        "num_inputs": len(texts),
        "num_ablations": num_ablations,
        "batch_results": batch_results,
    }


@app.local_entrypoint()
def main():
    """
    Example usage of Modal functions.
    Run with: modal run modal_app.py
    """
    # Example: Run attribution on a fine-tuned model
    # Replace with your actual model path
    model_path = "microsoft/deberta-v3-base"  # Or path to your fine-tuned model
    text = "Could you please help me with this issue when you have time?"
    
    print("Running attribution analysis...")
    results = run_attribution_analysis.remote(
        model_path=model_path,
        text=text,
        num_ablations=512,  # TA建议
        context_keep_prob=0.8,
        min_context_sentences=1,
        random_seed=42,
    )
    
    print("\nResults:")
    print(f"Baseline prediction: {results['baseline_prediction']['label']}")
    print("\nTop 3 most important sentences:")
    for i, sent in enumerate(results['sentences'][:3], 1):
        print(f"{i}. Score: {sent['attribution_score']:.4f}")
        print(f"   {sent['sentence'][:80]}...")