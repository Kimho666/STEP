#!/usr/bin/env python
"""
Summarize multi-seed eval_results_comparison.json files into one CSV.

Expected layout example:
  <root>/eval_combined_lift_3seeds/eval_results_comparison.json
  <root>/eval_combined_can_3seeds/eval_results_comparison.json

Output rows include:
- Action Predictor only
- Combined (per step)
- Diffusion only (per step)

For each row, this script exports:
- formatted score: 0.1034±0.0018
- numeric mean and std columns
"""

import argparse
import json
import math
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isfinite(f):
            return f
        return None
    try:
        f = float(v)
        if math.isfinite(f):
            return f
        return None
    except Exception:
        return None


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    n = len(values)
    mean_v = sum(values) / n
    var_v = sum((x - mean_v) ** 2 for x in values) / n
    std_v = var_v ** 0.5
    return mean_v, std_v


def _format_mean_std(mean_v: Optional[float], std_v: Optional[float]) -> str:
    if mean_v is None:
        return ""
    if std_v is None:
        std_v = 0.0
    return f"{mean_v:.4f}±{std_v:.4f}"


def _extract_metric(
    metrics: Dict[str, Any],
    per_seed_metrics: Optional[List[Dict[str, Any]]],
    metric_key: str,
) -> Tuple[str, Optional[float], Optional[float], int]:
    """
    Return (formatted, mean, std, n_seeds).
    Priority:
    1) metric_key_formatted in metrics
    2) compute from per_seed_metrics
    3) metric_key in metrics with std=0
    """
    formatted_key = f"{metric_key}_formatted"
    std_key = f"{metric_key}_std"

    raw_mean = _to_float(metrics.get(metric_key))
    raw_std = _to_float(metrics.get(std_key))

    if isinstance(metrics.get(formatted_key), str) and metrics.get(formatted_key):
        formatted = metrics[formatted_key]
        n_seeds = int(_to_float(metrics.get("num_seeds")) or 0)
        return formatted, raw_mean, raw_std, n_seeds

    if per_seed_metrics:
        vals: List[float] = []
        for item in per_seed_metrics:
            if not isinstance(item, dict):
                continue
            fv = _to_float(item.get(metric_key))
            if fv is not None:
                vals.append(fv)
        mean_v, std_v = _mean_std(vals)
        return _format_mean_std(mean_v, std_v), mean_v, std_v, len(vals)

    if raw_mean is not None:
        std_v = raw_std if raw_std is not None else 0.0
        return _format_mean_std(raw_mean, std_v), raw_mean, std_v, int(_to_float(metrics.get("num_seeds")) or 1)

    return "", None, None, 0


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def summarize_one_json(json_path: pathlib.Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    exp_dir = json_path.parent
    exp_name = exp_dir.name
    rows: List[Dict[str, Any]] = []

    # Action Predictor only
    ap_metrics = data.get("action_predictor_only", {})
    ap_per_seed = data.get("action_predictor_only_per_seed", [])
    test_fmt, test_mean, test_std, n_seeds = _extract_metric(ap_metrics, ap_per_seed, "test/mean_score")
    train_fmt, train_mean, train_std, _ = _extract_metric(ap_metrics, ap_per_seed, "train/mean_score")

    rows.append(
        {
            "experiment": exp_name,
            "json_path": str(json_path),
            "policy": "action_predictor_only",
            "steps": "",
            "test_score": test_fmt,
            "test_mean": test_mean,
            "test_std": test_std,
            "train_score": train_fmt,
            "train_mean": train_mean,
            "train_std": train_std,
            "num_seeds": n_seeds,
        }
    )

    # Steps entries
    for key, value in data.items():
        if not isinstance(key, str) or not key.startswith("steps_"):
            continue

        try:
            steps = int(key.split("_", 1)[1])
        except Exception:
            continue

        # Combined
        combined_metrics = _safe_get(value, ["combined", "metrics"], {}) or {}
        combined_per_seed = value.get("combined_per_seed_metrics", [])
        test_fmt, test_mean, test_std, n_seeds = _extract_metric(combined_metrics, combined_per_seed, "test/mean_score")
        train_fmt, train_mean, train_std, _ = _extract_metric(combined_metrics, combined_per_seed, "train/mean_score")

        rows.append(
            {
                "experiment": exp_name,
                "json_path": str(json_path),
                "policy": "combined",
                "steps": steps,
                "test_score": test_fmt,
                "test_mean": test_mean,
                "test_std": test_std,
                "train_score": train_fmt,
                "train_mean": train_mean,
                "train_std": train_std,
                "num_seeds": n_seeds,
            }
        )

        # Diffusion only
        diffusion_metrics = _safe_get(value, ["diffusion_only", "metrics"], {}) or {}
        diffusion_per_seed = value.get("diffusion_only_per_seed_metrics", [])
        test_fmt, test_mean, test_std, n_seeds = _extract_metric(diffusion_metrics, diffusion_per_seed, "test/mean_score")
        train_fmt, train_mean, train_std, _ = _extract_metric(diffusion_metrics, diffusion_per_seed, "train/mean_score")

        rows.append(
            {
                "experiment": exp_name,
                "json_path": str(json_path),
                "policy": "diffusion_only",
                "steps": steps,
                "test_score": test_fmt,
                "test_mean": test_mean,
                "test_std": test_std,
                "train_score": train_fmt,
                "train_mean": train_mean,
                "train_std": train_std,
                "num_seeds": n_seeds,
            }
        )

    return rows


def find_json_files(input_root: pathlib.Path, pattern: str) -> List[pathlib.Path]:
    return sorted(input_root.rglob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Summarize multi-seed evaluation JSON files into CSV")
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing multiple eval folders",
    )
    parser.add_argument(
        "--pattern",
        default="eval_results_comparison.json",
        help="Filename pattern to search recursively",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (.xlsx/.xls for Excel, others for CSV)",
    )
    args = parser.parse_args()

    input_root = pathlib.Path(args.input_root).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()
    if output_path.suffix == "":
        # 默认导出 Excel，便于直接做表格分析
        output_path = output_path.with_suffix(".xlsx")

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    json_files = find_json_files(input_root, args.pattern)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {input_root} with pattern {args.pattern}")

    all_rows: List[Dict[str, Any]] = []
    for json_path in json_files:
        all_rows.extend(summarize_one_json(json_path))

    if not all_rows:
        raise RuntimeError("No rows parsed from JSON files")

    df = pd.DataFrame(all_rows)
    df.sort_values(by=["experiment", "policy", "steps"], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Parsed JSON files: {len(json_files)}")
    print(f"Total rows: {len(df)}")
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
