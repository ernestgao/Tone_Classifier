from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import DatasetDict, load_dataset

LABEL_TO_ID = {
    "impolite": 0,
    "neutral": 1,
    "polite": 2,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Supports Stanford API style labels: -1 impolite, 0 neutral, 1 polite.
INT_LABEL_MAP = {
    -1: LABEL_TO_ID["impolite"],
    0: LABEL_TO_ID["neutral"],
    1: LABEL_TO_ID["polite"],
}

STR_LABEL_MAP = {
    "-1": LABEL_TO_ID["impolite"],
    "0": LABEL_TO_ID["neutral"],
    "1": LABEL_TO_ID["polite"],
    "impolite": LABEL_TO_ID["impolite"],
    "neutral": LABEL_TO_ID["neutral"],
    "polite": LABEL_TO_ID["polite"],
}


@dataclass
class DataConfig:
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"


def _guess_loader(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(f"Unsupported file extension for {path}. Use csv/json/jsonl/parquet.")


def load_politeness_dataset(cfg: DataConfig) -> DatasetDict:
    if cfg.dataset_name:
        dataset = load_dataset(cfg.dataset_name, cfg.dataset_config_name)
    elif cfg.train_file:
        data_files = {"train": cfg.train_file}
        if cfg.validation_file:
            data_files["validation"] = cfg.validation_file
        if cfg.test_file:
            data_files["test"] = cfg.test_file
        dataset = load_dataset(_guess_loader(cfg.train_file), data_files=data_files)
    else:
        raise ValueError(
            "Provide either --dataset_name (Hugging Face dataset) or --train_file (local file)."
        )

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
    if "test" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=17)
        dataset = DatasetDict(
            {
                "train": split["train"],
                "validation": dataset["validation"],
                "test": split["test"],
            }
        )

    needed = {cfg.text_column, cfg.label_column}
    for split_name in ["train", "validation", "test"]:
        missing = needed - set(dataset[split_name].column_names)
        if missing:
            raise ValueError(
                f"Split '{split_name}' is missing required columns: {sorted(missing)}."
            )

    return dataset


def normalize_label(raw_label) -> int:
    if isinstance(raw_label, bool):
        raw_label = int(raw_label)

    if isinstance(raw_label, int):
        if raw_label in INT_LABEL_MAP:
            return INT_LABEL_MAP[raw_label]
        raise ValueError(f"Unsupported integer label: {raw_label}. Expected one of -1, 0, 1.")

    if isinstance(raw_label, float):
        int_label = int(raw_label)
        if int_label == raw_label and int_label in INT_LABEL_MAP:
            return INT_LABEL_MAP[int_label]
        raise ValueError(f"Unsupported float label: {raw_label}. Expected exactly -1, 0, 1.")

    label_key = str(raw_label).strip().lower()
    if label_key in STR_LABEL_MAP:
        return STR_LABEL_MAP[label_key]

    raise ValueError(
        f"Unsupported label '{raw_label}'. Expected one of -1/0/1 or polite/neutral/impolite."
    )


def prepare_dataset(
    dataset: DatasetDict,
    tokenizer,
    text_column: str,
    label_column: str,
    max_length: int,
) -> DatasetDict:
    def _transform(batch: Dict) -> Dict:
        enc = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = [normalize_label(x) for x in batch[label_column]]
        return enc

    processed = dataset.map(
        _transform,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing and mapping labels",
    )
    return processed


def class_weights_from_dataset(dataset: DatasetDict) -> Tuple[float, float, float]:
    counts = [1, 1, 1]  # light smoothing to avoid zero division edge cases
    for label in dataset["train"]["labels"]:
        counts[label] += 1

    total = float(sum(counts))
    weights = [total / (3.0 * c) for c in counts]
    return tuple(weights)
