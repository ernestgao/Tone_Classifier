from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run lightweight grid search for politeness classifier")
    p.add_argument("--python", type=str, default="python3")
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--validation_file", type=str, default=None)
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--label_column", type=str, default="label")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--base_output_dir", type=str, default="artifacts/tuning")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_one(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def main() -> None:
    args = parse_args()
    out_root = Path(args.base_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Balanced search space: strong quality without taking too long on 1 GPU.
    learning_rates = [1.5e-5, 2e-5]
    batch_sizes = [16, 32]
    epochs = [4, 5]
    weight_decays = [0.01]

    combos = list(itertools.product(learning_rates, batch_sizes, epochs, weight_decays))
    summary = []

    for i, (lr, bs, ep, wd) in enumerate(combos, start=1):
        exp_dir = out_root / f"exp_{i:02d}"
        cmd = [
            args.python,
            "-m",
            "politeness_classifier.train",
            "--model_name",
            "roberta-base",
            "--output_dir",
            str(exp_dir),
            "--learning_rate",
            str(lr),
            "--per_device_train_batch_size",
            str(bs),
            "--per_device_eval_batch_size",
            str(max(bs, 32)),
            "--num_train_epochs",
            str(ep),
            "--weight_decay",
            str(wd),
            "--warmup_ratio",
            "0.1",
            "--max_length",
            str(args.max_length),
            "--eval_steps",
            "200",
            "--logging_steps",
            "50",
            "--seed",
            str(args.seed),
        ]

        if args.dataset_name:
            cmd.extend(["--dataset_name", args.dataset_name])
        if args.dataset_config_name:
            cmd.extend(["--dataset_config_name", args.dataset_config_name])
        if args.train_file:
            cmd.extend(["--train_file", args.train_file])
        if args.validation_file:
            cmd.extend(["--validation_file", args.validation_file])
        if args.test_file:
            cmd.extend(["--test_file", args.test_file])

        cmd.extend(["--text_column", args.text_column, "--label_column", args.label_column])

        if args.fp16:
            cmd.append("--fp16")
        if args.use_class_weights:
            cmd.append("--use_class_weights")

        code = run_one(cmd)
        metric_file = exp_dir / "metrics.json"
        row = {
            "exp": i,
            "return_code": code,
            "learning_rate": lr,
            "batch_size": bs,
            "epochs": ep,
            "weight_decay": wd,
            "output_dir": str(exp_dir),
        }

        if code == 0 and metric_file.exists():
            with metric_file.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            row["val_accuracy"] = metrics.get("validation", {}).get("eval_accuracy")
            row["val_macro_f1"] = metrics.get("validation", {}).get("eval_macro_f1")
            row["test_accuracy"] = metrics.get("test", {}).get("test_accuracy")
            row["test_macro_f1"] = metrics.get("test", {}).get("test_macro_f1")
        summary.append(row)

        with (out_root / "tuning_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    successful = [r for r in summary if r.get("val_macro_f1") is not None]
    if successful:
        best = max(successful, key=lambda r: r["val_macro_f1"])
        print("Best experiment:")
        print(json.dumps(best, indent=2))
    else:
        print("No successful experiments.")


if __name__ == "__main__":
    main()
