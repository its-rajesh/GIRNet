from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


METRICS = (
    "sisdr_in", "sisdr_out", "delta_sisdr",
    "sir_in", "sir_out", "delta_sir", "sar_out",
)


def row_key(row: dict) -> tuple:
    return (
        str(row["track"]), int(row["start_sample"]), float(row["bleed_level_db"]),
        float(row.get("rt60_seconds", -1.0)), str(row.get("room_id", "")),
        int(row.get("graph_pattern_id", -1)), int(row["channel"]), str(row["stem"]),
    )


def mixture_key(row: dict) -> tuple:
    return (
        str(row["track"]), int(row["start_sample"]), float(row["bleed_level_db"]),
        float(row.get("rt60_seconds", -1.0)), str(row.get("room_id", "")),
        int(row.get("graph_pattern_id", -1)),
    )


def groups(rows: list[dict]) -> dict[str, list[dict]]:
    output = {"overall": rows}
    if rows and "source_subset" in rows[0]:
        for subset in sorted({str(row["source_subset"]) for row in rows}):
            output[f"subset/{subset}"] = [
                row for row in rows if str(row["source_subset"]) == subset
            ]
    for stem in sorted({str(row["stem"]) for row in rows}):
        output[f"stem/{stem}"] = [row for row in rows if str(row["stem"]) == stem]
    for level in sorted({float(row["bleed_level_db"]) for row in rows}):
        output[f"level/{level:g}dB"] = [row for row in rows if float(row["bleed_level_db"]) == level]
    if rows and "rt60_seconds" in rows[0]:
        for rt60 in sorted({float(row["rt60_seconds"]) for row in rows}):
            output[f"rt60/{rt60:g}s"] = [row for row in rows if float(row["rt60_seconds"]) == rt60]
        for stem in sorted({str(row["stem"]) for row in rows}):
            for level in sorted({float(row["bleed_level_db"]) for row in rows}):
                output[f"stem_level/{stem}/{level:g}dB"] = [
                    row for row in rows
                    if str(row["stem"]) == stem and float(row["bleed_level_db"]) == level
                ]
    return output


def _draws(track_names: list[str], replicates: int, seed: int) -> np.ndarray:
    if replicates < 10_000:
        raise ValueError("Track-cluster bootstrap requires at least 10,000 replicates")
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(track_names), size=(replicates, len(track_names)))


def _cluster_summary(
    selected: list[dict], metric: str, track_names: list[str], draws: np.ndarray,
    values_override: dict[tuple, float] | None = None,
) -> dict[str, float | int]:
    sums = np.zeros(len(track_names), dtype=np.float64)
    counts = np.zeros(len(track_names), dtype=np.int64)
    track_index = {track: index for index, track in enumerate(track_names)}
    all_values = []
    for row in selected:
        value = float(row[metric]) if values_override is None else float(values_override[row_key(row)])
        index = track_index[str(row["track"])]
        sums[index] += value
        counts[index] += 1
        all_values.append(value)
    if np.any(counts == 0):
        missing = [track_names[index] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f"Group lacks rows for tracks: {missing[:5]}")
    replicate_values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(replicate_values, [2.5, 97.5])
    return {
        "mean": float(np.mean(all_values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_std": float(replicate_values.std(ddof=1)),
        "independent_songs": len(track_names),
        "mixtures": len({mixture_key(row) for row in selected}),
        "source_estimates": len(selected),
    }


def track_cluster_bootstrap(rows: list[dict], replicates: int = 20_000, seed: int = 1337) -> dict:
    track_names = sorted({str(row["track"]) for row in rows})
    draws = _draws(track_names, replicates, seed)
    result = {
        "method": "complete-song percentile cluster bootstrap",
        "confidence_level": 0.95,
        "replicates": int(replicates),
        "seed": int(seed),
        "groups": {},
    }
    for name, selected in groups(rows).items():
        result["groups"][name] = {
            metric: _cluster_summary(selected, metric, track_names, draws)
            for metric in METRICS
        }
    return result


def paired_track_cluster_bootstrap(
    candidate_rows: list[dict], reference_rows: list[dict],
    replicates: int = 20_000, seed: int = 1337,
) -> dict:
    candidate = {row_key(row): row for row in candidate_rows}
    reference = {row_key(row): row for row in reference_rows}
    if candidate.keys() != reference.keys():
        missing_candidate = list(reference.keys() - candidate.keys())[:5]
        missing_reference = list(candidate.keys() - reference.keys())[:5]
        raise ValueError(
            f"Paired evaluations do not contain identical source estimates; "
            f"missing_candidate={missing_candidate}, missing_reference={missing_reference}"
        )
    ordered_rows = [candidate[key] for key in sorted(candidate)]
    track_names = sorted({str(row["track"]) for row in ordered_rows})
    draws = _draws(track_names, replicates, seed)
    differences = {
        metric: {
            key: float(candidate[key][metric]) - float(reference[key][metric])
            for key in candidate
        }
        for metric in METRICS
    }
    result = {
        "method": "paired complete-song percentile cluster bootstrap",
        "difference_direction": "candidate_minus_reference",
        "confidence_level": 0.95,
        "replicates": int(replicates),
        "seed": int(seed),
        "groups": {},
    }
    for name, selected in groups(ordered_rows).items():
        result["groups"][name] = {
            metric: _cluster_summary(selected, metric, track_names, draws, differences[metric])
            for metric in METRICS
        }
    return result


def read_metric_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def flatten_summary(summary: dict) -> list[dict]:
    rows = []
    for group, metrics in summary["groups"].items():
        for metric, values in metrics.items():
            rows.append({"group": group, "metric": metric, **values})
    return rows


def write_summary_csv(path: str | Path, summary: dict) -> None:
    rows = flatten_summary(summary)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
