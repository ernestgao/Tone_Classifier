from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Stanford politeness corpus via ConvoKit and create train/valid/test CSVs.")
    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--valid_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _map_binary_label(v) -> int:
    # ConvoKit/Stanford binary labels already use: 1 polite, 0 neutral, -1 impolite.
    if v in (-1, 0, 1):
        return int(v)
    try:
        iv = int(v)
        if iv in (-1, 0, 1):
            return iv
    except Exception:
        pass
    raise ValueError(f"Unexpected binary label value: {v}")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from convokit import Corpus, download
    except Exception as e:
        raise RuntimeError(
            "ConvoKit is required. Install with: pip install convokit"
        ) from e

    # This is Stanford's Wikipedia politeness corpus in ConvoKit.
    corpus_path = download("wikipedia-politeness-corpus")
    corpus = Corpus(filename=corpus_path)

    rows = []
    for utt in corpus.iter_utterances():
        text = getattr(utt, "text", None)
        meta = getattr(utt, "meta", {}) or {}

        # ConvoKit exposes these in this corpus.
        if "Binary" not in meta:
            continue
        if not text or not str(text).strip():
            continue

        label = _map_binary_label(meta["Binary"])
        rows.append({"text": str(text).strip(), "label": label})

    if not rows:
        raise RuntimeError(
            "No rows extracted. Check corpus metadata fields for your ConvoKit version."
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Stratified split: train / valid / test
    train_df, temp_df = train_test_split(
        df,
        test_size=(args.valid_size + args.test_size),
        random_state=args.seed,
        stratify=df["label"],
    )
    valid_ratio_in_temp = args.valid_size / (args.valid_size + args.test_size)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - valid_ratio_in_temp),
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    train_path = out / "train.csv"
    valid_path = out / "valid.csv"
    test_path = out / "test.csv"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train: {train_path} ({len(train_df)} rows)")
    print(f"Saved valid: {valid_path} ({len(valid_df)} rows)")
    print(f"Saved test : {test_path} ({len(test_df)} rows)")
    print("Label counts (full set):")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
