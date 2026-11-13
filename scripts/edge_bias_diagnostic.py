#!/usr/bin/env python3
"""Descriptive edge-bias diagnostic for saved heterogeneous graph predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prediction_db(values: np.ndarray, floor_db: float) -> np.ndarray:
    floor_amplitude = 10.0 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(values.astype(np.float64), floor_amplitude))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("paper_spl/convolutive/heterogeneous_v1/test.jsonl"),
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=Path(
            "paper_spl/convolutive/heterogeneous_v1/evaluations/"
            "girnet_bf16/predicted_adjacency.npy"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "paper_spl/convolutive/heterogeneous_v1/evaluations/"
            "girnet_bf16/graph_interpretability"
        ),
    )
    parser.add_argument("--db-floor", type=float, default=-60.0)
    args = parser.parse_args()

    entries = read_jsonl(args.manifest)
    predictions = np.load(args.predictions)
    if predictions.shape != (len(entries), 4, 4):
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match "
            f"{len(entries)} four-source manifest rows"
        )
    if not all("edge_levels_db" in entry for entry in entries):
        raise ValueError("Every manifest row must contain the true edge_levels_db matrix")

    mask = ~np.eye(4, dtype=bool)
    true_db = np.asarray(
        [entry["edge_levels_db"] for entry in entries], dtype=np.float64
    )[:, mask].reshape(-1)
    predicted_db = prediction_db(predictions[:, mask], args.db_floor).reshape(-1)
    error_db = predicted_db - true_db

    rows: list[dict[str, float | int]] = []
    for level in sorted(np.unique(true_db).tolist()):
        keep = true_db == level
        level_error = error_db[keep]
        level_prediction = predicted_db[keep]
        rows.append({
            "true_level_db": float(level),
            "count": int(keep.sum()),
            "mean_signed_error_db": float(level_error.mean()),
            "mae_db": float(np.abs(level_error).mean()),
            "error_std_db": float(level_error.std(ddof=1)),
            "predicted_mean_db": float(level_prediction.mean()),
            "predicted_median_db": float(np.median(level_prediction)),
            "predicted_std_db": float(level_prediction.std(ddof=1)),
        })

    slope, intercept = np.polyfit(true_db, predicted_db, 1)
    fitted = slope * true_db + intercept
    residual_sum_squares = np.square(predicted_db - fitted).sum()
    total_sum_squares = np.square(predicted_db - predicted_db.mean()).sum()
    overall = {
        "directed_edge_estimates": int(true_db.size),
        "mixtures": len(entries),
        "songs": len({entry["track"] for entry in entries}),
        "db_floor": args.db_floor,
        "mean_signed_error_db": float(error_db.mean()),
        "mae_db": float(np.abs(error_db).mean()),
        "error_std_db": float(error_db.std(ddof=1)),
        "predicted_mean_db": float(predicted_db.mean()),
        "predicted_median_db": float(np.median(predicted_db)),
        "pearson_r_pooled": float(np.corrcoef(true_db, predicted_db)[0, 1]),
        "regression": {
            "dependent_variable": "predicted_edge_level_db",
            "independent_variable": "true_edge_level_db",
            "weighting": "ordinary least squares over all individual directed edges",
            "slope": float(slope),
            "intercept_db": float(intercept),
            "r_squared": float(1.0 - residual_sum_squares / total_sum_squares),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "edge_bias_by_true_level.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / "edge_bias_diagnostic.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "definition": "error_db = predicted_edge_level_db - true_edge_level_db",
            "calibration_policy": "descriptive only; no post-hoc fit is applied to predictions",
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "predictions": {"path": str(args.predictions), "sha256": sha256(args.predictions)},
            "overall": overall,
            "by_true_level": rows,
        }, handle, indent=2)
        handle.write("\n")

    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
