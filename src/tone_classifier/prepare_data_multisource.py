from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a multi-source politeness dataset and export train/valid/test CSVs "
            "with columns: text,label where label in {-1,0,1}."
        )
    )
    p.add_argument("--output_dir", type=str, default="data_multisource")
    p.add_argument("--valid_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # ConvoKit sources (human-annotated)
    p.add_argument("--use_wiki", action="store_true")
    p.add_argument("--use_stackexchange", action="store_true")

    # Optional local CSV source (e.g., PolitePEER export)
    p.add_argument(
        "--politepeer_csv",
        type=str,
        default=None,
        help=(
            "Optional local CSV path for PolitePEER-like 5-class data. "
            "Requires a text column and a label column."
        ),
    )
    p.add_argument("--politepeer_text_col", type=str, default="text")
    p.add_argument("--politepeer_label_col", type=str, default="label")
    p.add_argument(
        "--politepeer_mode",
        choices=["five_to_three", "extreme_only_three"],
        default="five_to_three",
        help=(
            "five_to_three: map {HI, I}->{-1}, N->{0}, {P, HP}->{1}; "
            "extreme_only_three: keep only {HI, N, HP} then map to {-1,0,1}."
        ),
    )

    # Optional HF dataset source (e.g., Intel/polite-guard)
    p.add_argument(
        "--polite_guard_dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset name (e.g., Intel/polite-guard).",
    )
    p.add_argument(
        "--polite_guard_text_col",
        type=str,
        default="text",
        help="Text column for polite-guard style dataset.",
    )
    p.add_argument(
        "--polite_guard_label_col",
        type=str,
        default="label",
        help="Label column for polite-guard style dataset.",
    )
    p.add_argument(
        "--somewhat_to",
        choices=["polite", "neutral"],
        default="neutral",
        help="Map 'somewhat polite' to +1 (polite) or 0 (neutral).",
    )
    p.add_argument(
        "--max_synth_fraction",
        type=float,
        default=0.3,
        help="Max synthetic-data fraction vs human data for final training pool.",
    )

    return p.parse_args()


def _map_binary_label(v: Any) -> int:
    if v in (-1, 0, 1):
        return int(v)
    try:
        iv = int(v)
        if iv in (-1, 0, 1):
            return iv
    except Exception:
        pass
    raise ValueError(f"Unexpected binary label value: {v}")


def _map_politepeer_label(v: Any, mode: str) -> int | None:
    """
    Map common 5-class PolitePEER-style labels to {-1,0,1}.
    Supports both numeric classes and string labels.
    """
    s = str(v).strip().lower()

    if mode not in {"five_to_three", "extreme_only_three"}:
        raise ValueError(f"Unknown politepeer mode: {mode}")

    # numeric style (commonly 1..5)
    try:
        k = int(s)
        if mode == "five_to_three":
            if k in (1, 2):
                return -1
            if k == 3:
                return 0
            if k in (4, 5):
                return 1
        else:
            # Keep only highly impolite / neutral / highly polite
            if k == 1:
                return -1
            if k == 3:
                return 0
            if k == 5:
                return 1
            return None
    except Exception:
        pass

    # string style
    if mode == "five_to_three":
        if "highly impolite" in s or s == "impolite":
            return -1
        if s == "neutral":
            return 0
        if "highly polite" in s or s == "polite":
            return 1
    else:
        # Keep only extreme labels plus neutral
        if "highly impolite" in s:
            return -1
        if s == "neutral":
            return 0
        if "highly polite" in s:
            return 1
        return None

    raise ValueError(f"Unknown PolitePEER label value: {v}")


def _map_polite_guard_label(v: Any, somewhat_to: str) -> int:
    s = str(v).strip().lower()

    if s in {"-1", "0", "1"}:
        return int(s)
    if s == "impolite":
        return -1
    if s == "neutral":
        return 0
    if s == "polite":
        return 1
    if s in {"somewhat polite", "somewhat_polite"}:
        return 1 if somewhat_to == "polite" else 0

    # Some datasets use integer class ids.
    try:
        iv = int(s)
        if iv in (-1, 0, 1):
            return iv
    except Exception:
        pass

    raise ValueError(f"Unknown polite-guard label value: {v}")


def _normalize_text_label_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["text", "label"])
    out["text"] = out["text"].astype(str).str.strip()
    out = out[~out["text"].str.lower().isin({"", "nan", "none"})]
    out["label"] = out["label"].astype(int)
    out = out[out["label"].isin([-1, 0, 1])]
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return out


def _load_convokit_source(dataset_name: str) -> pd.DataFrame:
    try:
        from convokit import Corpus, download
    except Exception as e:
        raise RuntimeError("ConvoKit is required. Install with: pip install convokit") from e

    # ConvoKit dataset keys can differ by version / naming style.
    # Try common aliases so users don't fail on minor naming differences.
    alias_map: dict[str, list[str]] = {
        "wikipedia-politeness-corpus": ["wikipedia-politeness-corpus"],
        "stackexchange-politeness-corpus": [
            "stack-exchange-politeness-corpus",
            "stackexchange-politeness-corpus",
        ],
    }
    candidates = alias_map.get(dataset_name, [dataset_name])

    corpus_path = None
    last_error: Exception | None = None
    for name in candidates:
        try:
            corpus_path = download(name)
            break
        except Exception as e:
            last_error = e
            continue

    if corpus_path is None:
        raise RuntimeError(
            f"Failed to download ConvoKit corpus for '{dataset_name}'. "
            f"Tried aliases: {candidates}"
        ) from last_error
    corpus = Corpus(filename=corpus_path)
    rows: list[dict[str, Any]] = []
    for utt in corpus.iter_utterances():
        text = getattr(utt, "text", None)
        meta = getattr(utt, "meta", {}) or {}
        if "Binary" not in meta:
            continue
        if not text or not str(text).strip():
            continue
        rows.append({"text": str(text).strip(), "label": _map_binary_label(meta["Binary"])})

    if not rows:
        raise RuntimeError(
            f"No rows extracted from {dataset_name}. "
            "Check corpus metadata fields for your ConvoKit version."
        )
    return _normalize_text_label_df(pd.DataFrame(rows))


def _load_politepeer_csv(path: str, text_col: str, label_col: str, mode: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"PolitePEER CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise KeyError(f"PolitePEER text column not found: {text_col}")
    if label_col not in df.columns:
        raise KeyError(f"PolitePEER label column not found: {label_col}")

    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    out["label"] = out["label"].apply(lambda v: _map_politepeer_label(v, mode=mode))
    out = out[out["label"].notna()]
    return _normalize_text_label_df(out)


def _load_hf_polite_guard(
    dataset_name: str,
    text_col: str,
    label_col: str,
    somewhat_to: str,
) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("datasets is required. Install with: pip install datasets") from e

    ds = load_dataset(dataset_name)
    rows: list[dict[str, Any]] = []
    for split_name in ds.keys():
        for ex in ds[split_name]:
            text = ex.get(text_col)
            label = ex.get(label_col)
            if text is None or label is None:
                continue
            rows.append(
                {
                    "text": str(text).strip(),
                    "label": _map_polite_guard_label(label, somewhat_to=somewhat_to),
                }
            )

    if not rows:
        raise RuntimeError(
            f"No rows extracted from HF dataset {dataset_name} with columns "
            f"text_col={text_col}, label_col={label_col}."
        )
    return _normalize_text_label_df(pd.DataFrame(rows))


def _safe_stratified_split(
    df: pd.DataFrame,
    valid_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp_size = valid_size + test_size
    if not (0.0 < temp_size < 1.0):
        raise ValueError("valid_size + test_size must be in (0, 1).")

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=seed,
        stratify=df["label"],
    )
    valid_ratio_in_temp = valid_size / temp_size
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - valid_ratio_in_temp),
        random_state=seed,
        stratify=temp_df["label"],
    )
    return train_df, valid_df, test_df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    human_parts: list[pd.DataFrame] = []
    synth_parts: list[pd.DataFrame] = []
    source_stats: dict[str, int] = {}

    if args.use_wiki:
        wiki_df = _load_convokit_source("wikipedia-politeness-corpus")
        human_parts.append(wiki_df)
        source_stats["wiki"] = len(wiki_df)

    if args.use_stackexchange:
        se_df = _load_convokit_source("stackexchange-politeness-corpus")
        human_parts.append(se_df)
        source_stats["stackexchange"] = len(se_df)

    if args.politepeer_csv:
        peer_df = _load_politepeer_csv(
            args.politepeer_csv,
            text_col=args.politepeer_text_col,
            label_col=args.politepeer_label_col,
            mode=args.politepeer_mode,
        )
        human_parts.append(peer_df)
        source_stats["politepeer_csv"] = len(peer_df)

    if args.polite_guard_dataset:
        guard_df = _load_hf_polite_guard(
            dataset_name=args.polite_guard_dataset,
            text_col=args.polite_guard_text_col,
            label_col=args.polite_guard_label_col,
            somewhat_to=args.somewhat_to,
        )
        synth_parts.append(guard_df)
        source_stats["polite_guard_dataset"] = len(guard_df)

    if not human_parts and not synth_parts:
        raise RuntimeError(
            "No sources enabled. Set one or more of: "
            "--use_wiki, --use_stackexchange, --politepeer_csv, --polite_guard_dataset."
        )

    human_df = (
        pd.concat(human_parts, ignore_index=True)
        if human_parts
        else pd.DataFrame(columns=["text", "label"])
    )
    human_df = _normalize_text_label_df(human_df) if not human_df.empty else human_df

    synth_df = (
        pd.concat(synth_parts, ignore_index=True)
        if synth_parts
        else pd.DataFrame(columns=["text", "label"])
    )
    synth_df = _normalize_text_label_df(synth_df) if not synth_df.empty else synth_df

    if not synth_df.empty and not human_df.empty:
        if not (0.0 <= args.max_synth_fraction <= 1.0):
            raise ValueError("--max_synth_fraction must be in [0,1].")
        max_synth_rows = int(args.max_synth_fraction * len(human_df))
        if max_synth_rows <= 0:
            synth_df = synth_df.iloc[0:0]
        elif len(synth_df) > max_synth_rows:
            synth_df = synth_df.sample(n=max_synth_rows, random_state=args.seed).reset_index(
                drop=True
            )

    final_df = pd.concat([human_df, synth_df], ignore_index=True)
    final_df = _normalize_text_label_df(final_df)
    if final_df.empty:
        raise RuntimeError("No rows left after filtering and deduplication.")

    train_df, valid_df, test_df = _safe_stratified_split(
        final_df,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    train_path = out_dir / "train.csv"
    valid_path = out_dir / "valid.csv"
    test_path = out_dir / "test.csv"
    stats_path = out_dir / "source_stats.json"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    human_rows = len(human_df)
    synth_rows = len(synth_df)
    full_stats = {
        "source_row_counts_before_merge": source_stats,
        "human_rows_after_dedup": human_rows,
        "synthetic_rows_after_cap": synth_rows,
        "total_rows_final": len(final_df),
        "label_counts_final": final_df["label"].value_counts().sort_index().to_dict(),
        "output_files": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
        },
        "config": {
            "seed": args.seed,
            "valid_size": args.valid_size,
            "test_size": args.test_size,
            "max_synth_fraction": args.max_synth_fraction,
            "somewhat_to": args.somewhat_to,
            "use_wiki": args.use_wiki,
            "use_stackexchange": args.use_stackexchange,
            "politepeer_csv": args.politepeer_csv,
            "politepeer_mode": args.politepeer_mode,
            "polite_guard_dataset": args.polite_guard_dataset,
        },
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(full_stats, f, indent=2, ensure_ascii=False)

    print(f"Saved train: {train_path} ({len(train_df)} rows)")
    print(f"Saved valid: {valid_path} ({len(valid_df)} rows)")
    print(f"Saved test : {test_path} ({len(test_df)} rows)")
    print(f"Saved stats: {stats_path}")
    print("Final label counts:")
    print(final_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()

