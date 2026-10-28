#!/usr/bin/env python3
"""Run independent training seeds on one fixed data manifest and summarize results."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def runtime_metadata():
    import torch
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in (
            "numpy", "pandas", "Pillow", "scikit-learn", "torch", "torchvision", "timm"
        )},
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_names": ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                      if cuda_available else []),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }

def aggregate_metric_files(run_dirs, filename, sections):
    per_seed = {}
    for run_dir in run_dirs:
        data = load_json(run_dir / filename)
        seed = str(data["run_seed"])
        per_seed[seed] = data
    summary = {}
    for section in sections:
        metric_names = sorted({
            key for d in per_seed.values()
            for key, value in d.get(section, {}).items()
            if isinstance(value, (int, float))
        })
        summary[section] = {}
        for metric in metric_names:
            values = [d[section][metric] for d in per_seed.values()
                      if metric in d.get(section, {})
                      and isinstance(d[section][metric], (int, float))]
            summary[section][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_runs": len(values),
            }
    return per_seed, summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["Kvasir", "GERD"])
    p.add_argument("--root", required=True, help="Dataset class-folder root")
    p.add_argument("--manifest", required=True, help="One fixed train/val/test manifest")
    p.add_argument("--config", default="config.json")
    p.add_argument("--out", required=True, help="Output root for all seeds")
    p.add_argument("--seeds", default="42,123,2026,3407,5678")
    p.add_argument("--pretrained", action="store_true", default=True)
    args = p.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        p.error("Provide at least two unique seeds")
    root, manifest, config, out = map(Path, (args.root, args.manifest, args.config, args.out))
    if not manifest.is_file():
        p.error(f"Manifest not found: {manifest}")
    if not config.is_file():
        p.error(f"Configuration not found: {config}")
    script = Path(__file__).with_name("gi_ensemble.py")
    if not script.is_file():
        p.error(f"Training script not found: {script}")
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f"seed_{seed}").exists() for seed in seeds):
        p.error("Output already contains one or more requested seed folders; choose an empty output folder")
    manifest_hash, config_hash = sha256(manifest), sha256(config)
    script_hash = sha256(script)
    runner_hash = sha256(Path(__file__))
    (out / "input_manifest.csv").write_bytes(manifest.read_bytes())
    (out / "config.json").write_bytes(config.read_bytes())
    metadata = {
        "dataset": args.dataset,
        "protocol": "fixed train/validation/test manifest; independent seeded training runs",
        "seeds": seeds,
        "manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "training_script_sha256": script_hash,
        "repeated_runner_sha256": runner_hash,
        "pretrained": bool(args.pretrained),
        "runtime": runtime_metadata(),
        "standard_deviation": "sample standard deviation (ddof=1)",
        "metric_units": "fraction (multiply by 100 for percentage reporting)",
    }
    (out / "reproduction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run_dirs = []
    for seed in seeds:
        base = out / f"seed_{seed}"
        base.mkdir(parents=True)
        models, features, results = base / "models", base / "features", base / "results"
        no_aug_models, no_aug_features = base / "models_no_aug", base / "features_no_aug"
        commands = [
            [sys.executable, str(script), "train-all", "--root", str(root), "--manifest", str(manifest),
             "--out", str(models), "--config", str(config), "--run-seed", str(seed)],
            [sys.executable, str(script), "extract-all", "--root", str(root), "--manifest", str(manifest),
             "--models", str(models), "--out", str(features), "--config", str(config), "--run-seed", str(seed)],
            [sys.executable, str(script), "train-all", "--root", str(root), "--manifest", str(manifest),
             "--out", str(no_aug_models), "--config", str(config), "--run-seed", str(seed), "--no-augment"],
            [sys.executable, str(script), "extract-all", "--root", str(root), "--manifest", str(manifest),
             "--models", str(no_aug_models), "--out", str(no_aug_features), "--config", str(config),
             "--run-seed", str(seed)],
            [sys.executable, str(script), "ensemble", "--features", str(features), "--out", str(results),
             "--config", str(config), "--run-seed", str(seed),
             "--no-aug-features", str(no_aug_features), "--dataset", args.dataset],
        ]
        if args.pretrained:
            commands[0].append("--pretrained")
            commands[2].append("--pretrained")
        command_records = []
        for index, command in enumerate(commands, start=1):
            log_path = base / f"step_{index:02d}.log"
            print(f"RUN seed={seed}: {' '.join(command)} (log: {log_path})", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            command_records.append({"argv": command, "log": log_path.name})
        (base / "commands.json").write_text(json.dumps({
            "seed": seed, "manifest_sha256": manifest_hash,
            "config_sha256": config_hash, "training_script_sha256": script_hash,
            "repeated_runner_sha256": runner_hash,
            "commands": command_records,
        }, indent=2), encoding="utf-8")
        run_dirs.append(results)

    ablations, ablation_summary = aggregate_metric_files(
        run_dirs, "ablation_metrics.json", ["A0", "A1", "A2", "A3", "A4", "A5"])
    table10_runs = {str(load_json(d / "table10_fusion_metrics.json")["run_seed"]):
                    load_json(d / "table10_fusion_metrics.json") for d in run_dirs}
    table10_summary = {}
    for strategy in ("feature_averaging", "feature_concatenation"):
        table10_summary[strategy] = {}
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            values = [d[strategy][metric] for d in table10_runs.values()]
            table10_summary[strategy][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "n_runs": len(values),
            }
    summary = {
        "protocol": "fixed manifest; independent model/projection/ensemble seeds",
        "seeds": seeds,
        "manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "training_script_sha256": script_hash,
        "ablation_summary": ablation_summary,
        "table10_summary": table10_summary,
        "per_seed_ablation_results": ablations,
        "per_seed_table10_results": table10_runs,
        "note": "Sample standard deviation uses ddof=1. Values are fractions; multiply by 100 for percentages.",
    }
    (out / "repeated_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_seed_rows, summary_rows = [], []
    for seed, data in ablations.items():
        for ablation, values in data.items():
            if ablation.startswith("A") and isinstance(values, dict):
                for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
                    if isinstance(values.get(metric), (int, float)):
                        per_seed_rows.append({"seed": seed, "configuration": ablation,
                                              "metric": metric, "value": values[metric]})
    for ablation, metrics_summary in ablation_summary.items():
        for metric, values in metrics_summary.items():
            summary_rows.append({"configuration": ablation, "metric": metric,
                                 "mean": values["mean"], "sample_sd": values["sample_sd"],
                                 "n_runs": values["n_runs"], "mean_percent": 100 * values["mean"],
                                 "sample_sd_percentage_points": 100 * values["sample_sd"]})
    for seed, data in table10_runs.items():
        for strategy in ("feature_averaging", "feature_concatenation"):
            for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
                per_seed_rows.append({"seed": seed, "configuration": strategy,
                                      "metric": metric, "value": data[strategy][metric]})
    for strategy, metric_summary in table10_summary.items():
        for metric, values in metric_summary.items():
            summary_rows.append({"configuration": strategy, "metric": metric,
                                 **values, "mean_percent": 100 * values["mean"],
                                 "sample_sd_percentage_points": 100 * values["sample_sd"]})
    write_csv(out / "per_seed_metrics.csv", per_seed_rows,
              ["seed", "configuration", "metric", "value"])
    write_csv(out / "mean_sample_sd.csv", summary_rows,
              ["configuration", "metric", "mean", "sample_sd", "n_runs",
               "mean_percent", "sample_sd_percentage_points"])
    print(json.dumps({"summary": str(out / "repeated_run_summary.json"),
                      "seeds": seeds, "manifest_sha256": summary["manifest_sha256"]}, indent=2))

def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
