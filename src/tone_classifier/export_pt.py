from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export HF sequence classifier to .pt")
    p.add_argument("--hf_model_dir", type=str, required=True)
    p.add_argument("--output_pt", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)

    output_pt = Path(args.output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "tokenizer_name_or_path": tokenizer.name_or_path,
    }
    torch.save(payload, output_pt)
    print(f"Saved {output_pt}")


if __name__ == "__main__":
    main()
