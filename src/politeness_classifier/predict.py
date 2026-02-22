from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_dir)
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(
            args.text,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred = int(torch.argmax(probs).item())

    print("label:", ID_TO_LABEL[pred])
    print("probabilities:")
    for i, p in enumerate(probs.tolist()):
        print(f"  {ID_TO_LABEL[i]}: {p:.4f}")


if __name__ == "__main__":
    main()
