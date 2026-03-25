#!/usr/bin/env python
"""
Summarize eval_combined_inference.py results.
Reads eval_results_comparison.json and outputs CSV/pretty table
with per-step metrics for combined and diffusion-only.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd


def safe_get(d: Dict[str, Any], keys: Tuple[str, ...], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def to_float(x: Any):
    """Best-effort float conversion; returns None if not convertible."""
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            return float(x)
    except Exception:
        return None
    return None


def mean_rewards(metrics: Dict[str, Any], prefix: str) -> Any:
    # Prefer aggregated mean_score if present
    direct = to_float(metrics.get(f"{prefix}/mean_score"))
    if direct is not None:
        return direct
    # Otherwise average all sim_max_reward_* entries
    vals = []
    for k, v in metrics.items():
        if k.startswith(f"{prefix}/sim_max_reward_"):
            fv = to_float(v)
            if fv is not None:
                vals.append(fv)
    if vals:
        return sum(vals) / len(vals)
    return None


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, value in data.items():
        if not key.startswith("steps_"):
            continue
        try:
            steps = int(key.split("_")[1])
        except Exception:
            continue
        combined_metrics = safe_get(value, ("combined", "metrics"), {}) or {}
        combined_timing = safe_get(value, ("combined", "timing"), {}) or {}
        diffusion_metrics = safe_get(value, ("diffusion_only", "metrics"), {}) or {}

        row = {
            "steps": steps,
            "combined_test": mean_rewards(combined_metrics, "test"),
            "combined_train": mean_rewards(combined_metrics, "train"),
            "diffusion_test": mean_rewards(diffusion_metrics, "test"),
            "diffusion_train": mean_rewards(diffusion_metrics, "train"),
            "combined_avg_time_ms": to_float(combined_timing.get("avg_total_time_ms")),
            "combined_ap_time_ms": to_float(combined_timing.get("avg_predictor_time_ms")),
            "combined_dp_time_ms": to_float(combined_timing.get("avg_diffusion_time_ms")),
            "speedup_ratio": to_float(combined_timing.get("speedup_ratio")),
            "total_inference_count": to_float(combined_timing.get("total_inference_count")),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["steps"])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize eval_combined_inference results")
    parser.add_argument("--input", required=True, help="Path to eval_results_comparison.json")
    parser.add_argument("--output", help="Optional CSV/Excel output path (.csv or .xlsx)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    data = load_results(args.input)
    rows = summarize(data)
    if not rows:
        print("No step entries found (steps_*) in JSON.")
        sys.exit(0)

    df = pd.DataFrame(rows)
    print("\nPer-step summary:\n")
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)
    print(df.to_string(index=False, formatters={col: fmt for col in df.columns}))

    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        if ext == ".xlsx":
            df.to_excel(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
