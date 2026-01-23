#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LLM SCALING ABLATION: 5 RUNS PER MODEL + MEAN±STD REPORT
================================================================================

Runs each LLM backbone (0.5B, 0.6B, 1.5B, 7B, 14B) for N seeds (default 5),
collects metrics from each run's `final_test_results.json`, and aggregates
mean ± std per model and across models.

Assumptions:
- TRAIN_SCRIPT writes: <run_dir>/final_test_results.json
- Your train script uses --seed, --language_model, --vision_model, etc.
- This script does NOT modify training code; it just orchestrates runs.

Outputs:
- For each model:
  OUTPUT_BASE/<model_key>/model_summary.json
- Overall:
  OUTPUT_BASE/ablation_summary.json

Usage:
------
python run_llm_scaling_ablation_5runs.py
python run_llm_scaling_ablation_5runs.py --num_runs 5 --base_seed 42
python run_llm_scaling_ablation_5runs.py --models 0.5B 1.5B 7B
python run_llm_scaling_ablation_5runs.py --dry_run

================================================================================
"""

import os
import re
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from statistics import mean, pstdev

# ==============================================================================
# USER CONFIG (as you provided)
# ==============================================================================

LLM_MODELS = {
    "0.5B": {
        "name": "Qwen2.5-0.5B",
        "path": "/mnt/share/ali/VLMs/hf_cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/",
        "params": "0.5B",
        "notes": "Smallest model - fast training, baseline comparison",
    },
    "0.6B": {
        "name": "Qwen3-0.6B",
        "path": "/mnt/share/ali/VLMs/hf_cache/hub/Qwen3-0.6B/",
        "params": "0.6B",
        "notes": "Current best model (84.82% accuracy) - REFERENCE",
    },
    "1.5B": {
        "name": "Qwen2.5-1.5B",
        "path": "/mnt/share/ali/Model_testing/models/Qwen/Qwen2.5-1.5B/",
        "params": "1.5B",
        "notes": "Medium model - balance of speed and capacity",
    },
    "7B": {
        "name": "Qwen2.5-7B",
        "path": "/mnt/share/ali/VLMs/hf_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796/",
        "params": "7B",
        "notes": "Large model - may need reduced batch size",
    },
    "14B": {
        "name": "Qwen2.5-14B",
        "path": "/mnt/share/ali/VLMs/hf_cache/hub/Qwen2___5-14B/",
        "params": "14B",
        "notes": "Extra large - highest capacity, slowest training",
    },
}

OPTIMAL_CONFIG = {
    "visual_tokens": 1024,
    "num_frames": 8,
    "lora_r": 32,
    "lora_alpha": 64,
    "learning_rate": 2e-4,
    "epochs": 20,
    "batch_size": 2,
    "grad_accum": 8,
    "seed": 42,
}

DATA_CONFIG = {
    "train_data": "/mnt/share/ali/VLM_Project/hospital_data/train.json",
    "val_data": "/mnt/share/ali/VLM_Project/hospital_data/val.json",
    "test_data": "/mnt/share/ali/VLM_Project/hospital_data/test.json",
    "vision_model": "/mnt/share/ali/VLMs/hf_cache/hub/siglip2-model/",
}

OUTPUT_BASE = "/mnt/share/ali/VLM_Project/ablation_results/ablation_llm_scale"
TRAIN_SCRIPT = "train_surgical_vlm_optimized.py"

BATCH_SIZE_CONFIG = {
    "0.5B": {"batch_size": 2, "grad_accum": 8},
    "0.6B": {"batch_size": 2, "grad_accum": 8},
    "1.5B": {"batch_size": 2, "grad_accum": 8},
    "7B": {"batch_size": 1, "grad_accum": 16},
    "14B": {"batch_size": 1, "grad_accum": 16},
}

# ==============================================================================
# HELPERS
# ==============================================================================

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def path_exists(p: str) -> bool:
    return Path(p).exists()

def safe_read_json(p: Path) -> Optional[dict]:
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None

def find_latest_run_dir(output_dir: Path) -> Optional[Path]:
    """
    Training script creates: output_dir/run_YYYYMMDD_HHMMSS
    Return latest run_*
    """
    if not output_dir.exists():
        return None
    runs = sorted(output_dir.glob("run_*"))
    return runs[-1] if runs else None

def build_train_cmd(model_key: str, seed: int, model_out_dir: Path) -> List[str]:
    """
    Build command to run one training job with a specific model+seed.
    We isolate per seed into: OUTPUT_BASE/<model_key>/seed_<seed>/
    """
    model_info = LLM_MODELS[model_key]
    bs = BATCH_SIZE_CONFIG.get(model_key, {"batch_size": OPTIMAL_CONFIG["batch_size"], "grad_accum": OPTIMAL_CONFIG["grad_accum"]})

    cmd = [
        "python", TRAIN_SCRIPT,
        "--train_data", DATA_CONFIG["train_data"],
        "--val_data", DATA_CONFIG["val_data"],
        "--test_data", DATA_CONFIG["test_data"],
        "--output_dir", str(model_out_dir),
        "--vision_model", DATA_CONFIG["vision_model"],
        "--language_model", model_info["path"],
        "--visual_tokens", str(OPTIMAL_CONFIG["visual_tokens"]),
        "--num_frames", str(OPTIMAL_CONFIG["num_frames"]),
        "--lora_r", str(OPTIMAL_CONFIG["lora_r"]),
        "--lora_alpha", str(OPTIMAL_CONFIG["lora_alpha"]),
        "--lr", str(OPTIMAL_CONFIG["learning_rate"]),
        "--epochs", str(OPTIMAL_CONFIG["epochs"]),
        "--batch_size", str(bs["batch_size"]),
        "--grad_accum", str(bs["grad_accum"]),
        "--seed", str(seed),
    ]
    return cmd

def is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def agg_mean_std(values: List[float]) -> Dict[str, float]:
    """
    Return mean and std (population std). If n=1, std=0.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "n": 1}
    return {"mean": float(mean(values)), "std": float(pstdev(values)), "n": len(values)}

def collect_numeric_metrics(run_metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten only numeric metrics from final_test_results.json.
    We keep floats/ints. Nested dicts are ignored.
    """
    out = {}
    for k, v in run_metrics.items():
        if isinstance(v, (int, float)) and is_number(v):
            out[k] = float(v)
    return out

def merge_metric_keys(list_of_dicts: List[Dict[str, float]]) -> List[str]:
    keys = set()
    for d in list_of_dicts:
        keys |= set(d.keys())
    return sorted(keys)

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

# ==============================================================================
# CORE
# ==============================================================================

def run_one(model_key: str, seed: int, base_out: Path, dry_run: bool, skip_existing: bool) -> Dict[str, Any]:
    """
    Runs training for one model+seed and returns metadata + metrics path.
    """
    seed_dir = base_out / model_key / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # if skipping and a previous run exists with final_test_results.json, skip
    latest = find_latest_run_dir(seed_dir)
    if skip_existing and latest and (latest / "final_test_results.json").exists():
        return {
            "status": "skipped",
            "model_key": model_key,
            "seed": seed,
            "seed_dir": str(seed_dir),
            "run_dir": str(latest),
            "final_test_results": str(latest / "final_test_results.json"),
        }

    cmd = build_train_cmd(model_key, seed, seed_dir)

    print("\n" + "=" * 90)
    print(f"MODEL={model_key} | SEED={seed}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 90)

    if dry_run:
        return {
            "status": "dry_run",
            "model_key": model_key,
            "seed": seed,
            "seed_dir": str(seed_dir),
            "cmd": cmd,
        }

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        t1 = time.time()

        latest = find_latest_run_dir(seed_dir)
        results_path = (latest / "final_test_results.json") if latest else None

        return {
            "status": "success",
            "model_key": model_key,
            "seed": seed,
            "seed_dir": str(seed_dir),
            "run_dir": str(latest) if latest else None,
            "final_test_results": str(results_path) if results_path else None,
            "wall_time_sec": float(t1 - t0),
        }
    except subprocess.CalledProcessError as e:
        t1 = time.time()
        return {
            "status": "failed",
            "model_key": model_key,
            "seed": seed,
            "seed_dir": str(seed_dir),
            "error": str(e),
            "return_code": e.returncode,
            "wall_time_sec": float(t1 - t0),
        }

def aggregate_model(model_key: str, base_out: Path, seeds: List[int]) -> Dict[str, Any]:
    """
    Aggregates all successful runs for a model across seeds.
    Reads final_test_results.json for each run.
    Produces mean±std for all numeric keys found.
    """
    model_dir = base_out / model_key
    runs_info = []
    metrics_list = []

    for seed in seeds:
        seed_dir = model_dir / f"seed_{seed}"
        latest = find_latest_run_dir(seed_dir)
        if not latest:
            runs_info.append({"seed": seed, "status": "missing"})
            continue

        results_path = latest / "final_test_results.json"
        if not results_path.exists():
            runs_info.append({"seed": seed, "status": "no_results", "run_dir": str(latest)})
            continue

        m = safe_read_json(results_path)
        if not isinstance(m, dict):
            runs_info.append({"seed": seed, "status": "bad_json", "run_dir": str(latest), "results": str(results_path)})
            continue

        flat = collect_numeric_metrics(m)
        runs_info.append({"seed": seed, "status": "ok", "run_dir": str(latest), "results": str(results_path)})
        metrics_list.append(flat)

    # aggregate
    keys = merge_metric_keys(metrics_list)
    agg = {}
    for k in keys:
        vals = [d[k] for d in metrics_list if k in d]
        agg[k] = agg_mean_std(vals)

    summary = {
        "model_key": model_key,
        "model_name": LLM_MODELS[model_key]["name"],
        "params": LLM_MODELS[model_key]["params"],
        "notes": LLM_MODELS[model_key]["notes"],
        "num_expected_runs": len(seeds),
        "num_success_runs": len(metrics_list),
        "runs": runs_info,
        "metrics_mean_std": agg,
    }
    return summary

def build_ablation_table(model_summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a compact table-like list for quick paper use.
    Tries to extract: accuracy, macro_f1, kappa, samples_per_sec, tokens_per_sec,
    total_params, trainable_params, training_seconds (if available).
    """
    table = []
    preferred = [
        "accuracy",
        "macro_f1",
        "kappa",
        "samples_per_sec",
        "tokens_per_sec",
        "total_params",
        "trainable_params",
        "training_seconds",
        "inference_seconds",
    ]

    for mk, summ in model_summaries.items():
        row = {
            "model_key": mk,
            "model_name": summ["model_name"],
            "params": summ["params"],
            "num_success_runs": summ["num_success_runs"],
        }
        ms = summ.get("metrics_mean_std", {})
        for k in preferred:
            if k in ms:
                row[k] = ms[k]
        table.append(row)

    # sort by param size order
    order = ["0.5B", "0.6B", "1.5B", "7B", "14B"]
    table.sort(key=lambda r: order.index(r["model_key"]) if r["model_key"] in order else 999)
    return table

# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run LLM Scaling Ablation: N runs per model + mean±std aggregation"
    )
    p.add_argument("--models", nargs="+", default=list(LLM_MODELS.keys()), choices=list(LLM_MODELS.keys()),
                   help="Which model keys to run")
    p.add_argument("--num_runs", type=int, default=5, help="Number of runs (seeds) per model")
    p.add_argument("--base_seed", type=int, default=42, help="Base seed (seeds will be base_seed..base_seed+num_runs-1)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Optional explicit seeds list (overrides num_runs/base_seed)")
    p.add_argument("--output_base", type=str, default=OUTPUT_BASE, help="Output base directory")
    p.add_argument("--dry_run", action="store_true", help="Print commands only")
    p.add_argument("--no_skip_existing", action="store_true", help="Re-run even if results already exist")
    p.add_argument("--results_only", action="store_true", help="Only aggregate existing results, do not run")
    return p.parse_args()

def main():
    args = parse_args()
    base_out = Path(args.output_base)
    base_out.mkdir(parents=True, exist_ok=True)

    # sanity check model paths
    print("\nChecking model paths:")
    available = []
    missing = []
    for mk in args.models:
        pth = LLM_MODELS[mk]["path"]
        if path_exists(pth):
            print(f"  ✓ {mk} {LLM_MODELS[mk]['name']}: {pth}")
            available.append(mk)
        else:
            print(f"  ✗ {mk} {LLM_MODELS[mk]['name']}: {pth}")
            missing.append(mk)

    if missing:
        print(f"\n⚠️ Missing models will be skipped: {missing}")

    models = available
    if not models:
        print("\n❌ No available models to run/aggregate.")
        return

    # seeds
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(args.base_seed, args.base_seed + args.num_runs))

    # Save run config
    run_config = {
        "timestamp": now_stamp(),
        "models": models,
        "seeds": seeds,
        "train_script": TRAIN_SCRIPT,
        "optimal_config": OPTIMAL_CONFIG,
        "data_config": DATA_CONFIG,
        "batch_size_config": BATCH_SIZE_CONFIG,
    }
    write_json(base_out / f"ablation_run_config_{run_config['timestamp']}.json", run_config)

    # run
    run_records = []
    if not args.results_only:
        for mk in models:
            for seed in seeds:
                rec = run_one(
                    model_key=mk,
                    seed=seed,
                    base_out=base_out,
                    dry_run=args.dry_run,
                    skip_existing=(not args.no_skip_existing),
                )
                run_records.append(rec)

        write_json(base_out / f"ablation_run_records_{run_config['timestamp']}.json", run_records)

        if args.dry_run:
            print("\n[DRY RUN] Done.")
            return

    # aggregate
    print("\nAggregating results...")
    model_summaries = {}
    for mk in models:
        summ = aggregate_model(mk, base_out, seeds)
        model_summaries[mk] = summ
        write_json(base_out / mk / "model_summary.json", summ)

    table = build_ablation_table(model_summaries)
    overall = {
        "timestamp": now_stamp(),
        "models": models,
        "seeds": seeds,
        "model_summaries": model_summaries,
        "table": table,
        "notes": (
            "metrics_mean_std contains mean/std/n for every numeric key found in final_test_results.json. "
            "If some metrics are missing, add them in train_surgical_vlm_optimized.py final_test_results.json."
        ),
    }
    write_json(base_out / "ablation_summary.json", overall)

    # Pretty print compact table
    print("\n" + "=" * 110)
    print("LLM SCALING ABLATION (MEAN ± STD over runs)")
    print("=" * 110)
    header = ["Model", "Params", "Runs", "Accuracy", "Macro-F1", "Kappa", "samples/s", "tokens/s"]
    print(f"{header[0]:<12} {header[1]:<8} {header[2]:<6} {header[3]:<18} {header[4]:<18} {header[5]:<18} {header[6]:<18} {header[7]:<18}")
    print("-" * 110)

    def fmt(ms, k):
        if k not in ms:
            return "NA"
        d = ms[k]
        return f"{d['mean']:.4f}±{d['std']:.4f}"

    for row in table:
        mk = row["model_key"]
        ms = model_summaries[mk].get("metrics_mean_std", {})
        acc = fmt(ms, "accuracy")
        f1 = fmt(ms, "macro_f1")
        kap = fmt(ms, "kappa")
        sps = fmt(ms, "samples_per_sec")
        tps = fmt(ms, "tokens_per_sec")
        print(f"{mk:<12} {row['params']:<8} {row['num_success_runs']:<6} {acc:<18} {f1:<18} {kap:<18} {sps:<18} {tps:<18}")

    print("=" * 110)
    print(f"\nSaved: {base_out / 'ablation_summary.json'}")
    for mk in models:
        print(f"Saved: {base_out / mk / 'model_summary.json'}")

if __name__ == "__main__":
    main()
