from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tuning search for tone classifier")
    p.add_argument("--python", type=str, default="python3")
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--validation_file", type=str, default=None)
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--label_column", type=str, default="label")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--base_output_dir", type=str, default="artifacts/tuning")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--profile", type=str, default="high_accuracy", choices=["fast", "balanced", "high_accuracy"])
    p.add_argument("--max_experiments", type=int, default=0, help="0 means run all experiments")
    return p.parse_args()


def run_one(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def profile_space(profile: str):
    if profile == "fast":
        models = ["microsoft/deberta-v3-base"]
        learning_rates = [2e-5]
        batch_sizes = [16]
        epochs = [4, 5]
        weight_decays = [0.01]
        grad_accums = [1]
        warmup_ratios = [0.1]
        seeds = [42]
        eval_steps = 200
    elif profile == "balanced":
        models = ["microsoft/deberta-v3-base"]
        learning_rates = [1e-5, 1.5e-5, 2e-5]
        batch_sizes = [16]
        epochs = [6, 8]
        weight_decays = [0.01, 0.05]
        grad_accums = [1, 2]
        warmup_ratios = [0.06, 0.1]
        seeds = [21, 42]
        eval_steps = 100
    else:
        # Best chance at >=75% while staying practical on one T4 session.
        models = ["microsoft/deberta-v3-base", "roberta-base"]
        learning_rates = [8e-6, 1e-5, 1.5e-5]
        batch_sizes = [16]
        epochs = [8, 10]
        weight_decays = [0.01, 0.05]
        grad_accums = [2]
        warmup_ratios = [0.06, 0.1]
        seeds = [13, 21, 42]
        eval_steps = 100

    return {
        "models": models,
        "learning_rates": learning_rates,
        "batch_sizes": batch_sizes,
        "epochs": epochs,
        "weight_decays": weight_decays,
        "grad_accums": grad_accums,
        "warmup_ratios": warmup_ratios,
        "seeds": seeds,
        "eval_steps": eval_steps,
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.base_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    space = profile_space(args.profile)
    combos = list(
        itertools.product(
            space["models"],
            space["learning_rates"],
            space["batch_sizes"],
            space["epochs"],
            space["weight_decays"],
            space["grad_accums"],
            space["warmup_ratios"],
            space["seeds"],
        )
    )

    if args.max_experiments > 0:
        combos = combos[: args.max_experiments]

    summary = []

    for i, (model_name, lr, bs, ep, wd, ga, wr, seed) in enumerate(combos, start=1):
        exp_dir = out_root / f"exp_{i:03d}"
        cmd = [
            args.python,
            "-m",
            "tone_classifier.train",
            "--model_name",
            str(model_name),
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
            str(wr),
            "--gradient_accumulation_steps",
            str(ga),
            "--max_length",
            str(args.max_length),
            "--eval_steps",
            str(space["eval_steps"]),
            "--logging_steps",
            "50",
            "--early_stopping_patience",
            "3",
            "--seed",
            str(seed),
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
            "model_name": model_name,
            "learning_rate": lr,
            "batch_size": bs,
            "epochs": ep,
            "weight_decay": wd,
            "grad_accum": ga,
            "warmup_ratio": wr,
            "seed": seed,
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
        ranked = sorted(
            successful,
            key=lambda r: (
                r.get("val_macro_f1", -1),
                r.get("val_accuracy", -1),
                r.get("test_macro_f1", -1),
            ),
            reverse=True,
        )
        best = ranked[0]
        with (out_root / "best_experiment.json").open("w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

        print("Best experiment:")
        print(json.dumps(best, indent=2))
        print("Top 5 by validation macro_f1:")
        print(json.dumps(ranked[:5], indent=2))
    else:
        print("No successful experiments.")


if __name__ == "__main__":
    main()
