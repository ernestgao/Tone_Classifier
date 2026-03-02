"""
Modal app for running attribution analysis on GPU.
Supports both gradient-based and non-gradient-based approaches.
"""
from typing import Any, Dict, List

import modal

# Define Modal app
app = modal.App("tone-classifier-attribution")
training_volume = modal.Volume.from_name("tone-classifier-artifacts", create_if_missing=True)

# Define image with all dependencies
# 参考示例：使用 image.add_local_dir() 而不是 Mount
# 注意：add_local_dir必须在最后，或者使用copy=True
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch>=2.1.0",
        "transformers>=4.42.0,<5.0.0",
        "accelerate>=0.31.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.20.0",
        "pandas>=2.0.0",
        "tiktoken",  # DeBERTa tokenizer需要
    ])
    .add_local_dir(
        ".",
        remote_path="/root/tone_classifier",
        ignore=["artifacts/", ".git/", "__pycache__/", "*.pyc", ".venv/"],
        copy=True,  # 复制文件到image中，这样可以在之后运行命令
    )
    .run_commands("cd /root/tone_classifier && pip install -e .")  # 安装tone_classifier包
)


def _resolve_remote_data_path(path_or_name: str) -> str:
    """
    Resolve data path for Modal runtime.
    Relative paths are assumed under /root/tone_classifier.
    """
    if path_or_name.startswith("/"):
        return path_or_name
    return f"/root/tone_classifier/{path_or_name}"


def _commit_training_volume_if_possible() -> None:
    """
    Commit writes for compatibility across Modal versions.
    """
    commit_fn = getattr(training_volume, "commit", None)
    if callable(commit_fn):
        commit_fn()


def _run_training_impl(
    model_name: str,
    train_file: str,
    validation_file: str,
    test_file: str,
    text_column: str,
    label_column: str,
    output_subdir: str,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    eval_steps: int,
    logging_steps: int,
    gradient_accumulation_steps: int,
    early_stopping_patience: int,
    max_length: int,
    dataloader_num_workers: int,
    seed: int,
    use_class_weights: bool,
    fp16: bool,
) -> Dict[str, Any]:
    import json
    import subprocess
    import time
    from pathlib import Path

    output_dir = f"/root/tone_classifier_outputs/{output_subdir}"
    train_file_path = _resolve_remote_data_path(train_file)
    valid_file_path = _resolve_remote_data_path(validation_file)
    test_file_path = _resolve_remote_data_path(test_file)

    cmd = [
        "python",
        "-u",
        "-m",
        "tone_classifier.train",
        "--train_file",
        train_file_path,
        "--validation_file",
        valid_file_path,
        "--test_file",
        test_file_path,
        "--text_column",
        text_column,
        "--label_column",
        label_column,
        "--model_name",
        model_name,
        "--max_length",
        str(max_length),
        "--num_train_epochs",
        str(num_train_epochs),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(per_device_eval_batch_size),
        "--learning_rate",
        str(learning_rate),
        "--weight_decay",
        str(weight_decay),
        "--warmup_ratio",
        str(warmup_ratio),
        "--eval_steps",
        str(eval_steps),
        "--logging_steps",
        str(logging_steps),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--early_stopping_patience",
        str(early_stopping_patience),
        "--dataloader_num_workers",
        str(dataloader_num_workers),
        "--seed",
        str(seed),
        "--output_dir",
        output_dir,
    ]
    if use_class_weights:
        cmd.append("--use_class_weights")
    if fp16:
        cmd.append("--fp16")

    print("Launching training command on Modal:")
    print(" ".join(cmd))
    start_time = time.time()
    completed = subprocess.run(
        cmd,
        cwd="/root/tone_classifier",
        text=True,
        capture_output=True,
    )
    elapsed = time.time() - start_time

    if completed.returncode != 0:
        raise RuntimeError(
            "Modal training subprocess failed.\n"
            f"return_code={completed.returncode}\n"
            f"stdout_tail:\n{completed.stdout[-6000:]}\n"
            f"stderr_tail:\n{completed.stderr[-6000:]}"
        )

    metrics_path = Path(output_dir) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    _commit_training_volume_if_possible()
    return {
        "ok": True,
        "elapsed_seconds": elapsed,
        "model_name": model_name,
        "output_dir": output_dir,
        "hf_model_dir": f"{output_dir}/hf_model",
        "metrics": metrics,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=21600,
    volumes={"/root/tone_classifier_outputs": training_volume},
)
def run_modal_training(
    model_name: str = "microsoft/deberta-v3-base",
    train_file: str = "data/train.csv",
    validation_file: str = "data/valid.csv",
    test_file: str = "data/test.csv",
    text_column: str = "text",
    label_column: str = "label",
    output_subdir: str = "deberta_modal_train",
    num_train_epochs: float = 8.0,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    eval_steps: int = 100,
    logging_steps: int = 10,
    gradient_accumulation_steps: int = 2,
    early_stopping_patience: int = 3,
    max_length: int = 128,
    dataloader_num_workers: int = 2,
    seed: int = 42,
    use_class_weights: bool = True,
    fp16: bool = True,
) -> Dict[str, Any]:
    """
    Train tone classifier on Modal and persist outputs to a Modal Volume.
    """
    return _run_training_impl(
        model_name=model_name,
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        text_column=text_column,
        label_column=label_column,
        output_subdir=output_subdir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        max_length=max_length,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        use_class_weights=use_class_weights,
        fp16=fp16,
    )


@app.function(
    image=image,
    gpu="H100:1",
    timeout=28800,
    volumes={"/root/tone_classifier_outputs": training_volume},
)
def run_modal_training_large(
    model_name: str = "microsoft/deberta-v3-large",
    train_file: str = "data/train.csv",
    validation_file: str = "data/valid.csv",
    test_file: str = "data/test.csv",
    text_column: str = "text",
    label_column: str = "label",
    output_subdir: str = "deberta_large_modal_train",
    num_train_epochs: float = 8.0,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 1.5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    eval_steps: int = 100,
    logging_steps: int = 10,
    gradient_accumulation_steps: int = 4,
    early_stopping_patience: int = 3,
    max_length: int = 128,
    dataloader_num_workers: int = 2,
    seed: int = 42,
    use_class_weights: bool = True,
    fp16: bool = True,
) -> Dict[str, Any]:
    """
    Larger-model training preset on Modal, intended for better accuracy.
    """
    return _run_training_impl(
        model_name=model_name,
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        text_column=text_column,
        label_column=label_column,
        output_subdir=output_subdir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        max_length=max_length,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        use_class_weights=use_class_weights,
        fp16=fp16,
    )


@app.function(
    image=image,
    gpu="A100-40GB",  # Use A100 for standard models (更新为新的API格式)
    timeout=3600,
    volumes={"/root/tone_classifier_outputs": training_volume},
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
    volumes={"/root/tone_classifier_outputs": training_volume},
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
    volumes={"/root/tone_classifier_outputs": training_volume},
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
    volumes={"/root/tone_classifier_outputs": training_volume},
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
    volumes={"/root/tone_classifier_outputs": training_volume},
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