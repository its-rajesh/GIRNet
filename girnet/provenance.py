from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import soundfile
import torch
import yaml


METRIC_DEFINITIONS = {
    "si_sdr": (
        "Ordered scale-invariant SDR. Target and estimate are independently mean-centered; "
        "the estimate is projected onto the fixed corresponding target; no PIT is used."
    ),
    "sir": (
        "Ordered global source-projection SIR. A full-segment least-squares projection onto "
        "all clean ordered sources defines target and interference components. This is not "
        "museval/BSS Eval v4."
    ),
    "sar": (
        "Ordered global source-projection SAR. The full four-source projection is compared "
        "with residual energy outside the clean-source subspace. This is not museval/BSS Eval v4."
    ),
    "improvement": "Calculated per source estimate as output metric minus contaminated-input metric before aggregation.",
    "adjacency": (
        "Twelve directed off-diagonal amplitudes are floored at -60 dB and compared in dB "
        "using Pearson correlation, MAE, and RMSE."
    ),
    "uncertainty": (
        "Percentile 95% confidence intervals from complete-song cluster bootstrap resampling; "
        "all crops, levels, and source estimates belonging to each sampled song are retained."
    ),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(project_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def software_versions() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "soundfile": soundfile.__version__,
        "pyyaml": yaml.__version__,
    }


def count_parameters(model: torch.nn.Module | None) -> int | None:
    if model is None:
        return None
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_evaluation_provenance(
    *,
    project_root: str | Path,
    config: dict,
    config_path: str | Path,
    manifest_path: str | Path,
    evaluation_script: str | Path,
    split: str,
    rows: list[dict],
    model: torch.nn.Module | None,
    model_mode: str | None,
    checkpoint_path: str | Path | None,
    predictions_path: str | Path | None,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    evaluation_script = Path(evaluation_script).resolve()
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
    predictions = Path(predictions_path).resolve() if predictions_path is not None else None
    tracks = {str(row["track"]) for row in rows}
    mixtures = {
        (str(row["track"]), int(row["start_sample"]), float(row["bleed_level_db"]))
        for row in rows
    }
    data = config["data"]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "git_commit": git_commit(project_root),
        "checkpoint": None if checkpoint is None else {
            "path": str(checkpoint), "sha256": file_sha256(checkpoint),
        },
        "external_predictions": None if predictions is None else {
            "path": str(predictions), "sha256": file_sha256(predictions),
        },
        "model": {
            "architecture_mode": model_mode,
            "parameter_count": count_parameters(model),
        },
        "configuration": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
            "snapshot": config,
        },
        "dataset": {
            "split": split,
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "source_order": list(data["stems"]),
            "sample_rate_hz": int(data["sample_rate"]),
            "segment_samples": int(data["segment_samples"]),
            "segment_duration_seconds": float(data["segment_samples"]) / float(data["sample_rate"]),
            "number_of_songs": len(tracks),
            "number_of_mixtures": len(mixtures),
            "number_of_source_estimates": len(rows),
        },
        "evaluation": {
            "script_path": str(evaluation_script),
            "script_sha256": file_sha256(evaluation_script),
            "metric_definitions": METRIC_DEFINITIONS,
            "bootstrap": {
                "unit": "song/track",
                "method": "percentile",
                "confidence_level": 0.95,
                "seed": int(bootstrap_seed),
                "replicates": int(bootstrap_replicates),
            },
        },
        "random_seed": int(config["seed"]),
        "software_versions": software_versions(),
    }


def save_provenance(path: str | Path, provenance: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
