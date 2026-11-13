#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from girnet.bootstrap import (
    paired_track_cluster_bootstrap, read_metric_rows, track_cluster_bootstrap, write_summary_csv,
)
from girnet.convolutive import ConvolutiveManifestDataset, ConvolutiveRenderDataset
from girnet.metrics import adjacency_metrics, bss_projection_metrics, si_sdr
from girnet.model import GIRNetSeparator, ModelConfig
from girnet.provenance import METRIC_DEFINITIONS, file_sha256, git_commit, software_versions
from girnet.utils import save_json, seed_everything


METRICS = ("sisdr_in", "sisdr_out", "delta_sisdr", "sir_in", "sir_out", "delta_sir", "sar_out")


def summarize(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std())}


def aggregate(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {"overall": rows}
    if rows and "source_subset" in rows[0]:
        for subset in sorted({row["source_subset"] for row in rows}):
            groups[f"subset/{subset}"] = [row for row in rows if row["source_subset"] == subset]
    for stem in sorted({row["stem"] for row in rows}):
        groups[f"stem/{stem}"] = [row for row in rows if row["stem"] == stem]
    for level in sorted({row["bleed_level_db"] for row in rows}):
        groups[f"level/{level:g}dB"] = [row for row in rows if row["bleed_level_db"] == level]
    for rt60 in sorted({row["rt60_seconds"] for row in rows}):
        groups[f"rt60/{rt60:g}s"] = [row for row in rows if row["rt60_seconds"] == rt60]
    for stem in sorted({row["stem"] for row in rows}):
        for level in sorted({row["bleed_level_db"] for row in rows}):
            groups[f"stem_level/{stem}/{level:g}dB"] = [
                row for row in rows if row["stem"] == stem and row["bleed_level_db"] == level
            ]
    return {name: {metric: summarize([float(row[metric]) for row in selected]) for metric in METRICS}
            for name, selected in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on a convolutive GIRNet manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1337)
    parser.add_argument("--reference-metrics-csv", type=Path)
    parser.add_argument("--stems", nargs="+", default=["vocals", "drums", "bass", "other"])
    parser.add_argument("--evaluation-scope")
    args = parser.parse_args()
    seed_everything(args.seed)
    with args.manifest.open(encoding="utf-8") as handle:
        first_entry = json.loads(next(line for line in handle if line.strip()))
    pre_rendered = "audio_path" in first_entry
    dataset = (
        ConvolutiveManifestDataset(args.manifest) if pre_rendered else
        ConvolutiveRenderDataset(args.manifest, stems=args.stems, include_contributions=True)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GIRNetSeparator(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    predicted_graphs, target_graphs = [], []
    inference_seconds = 0.0
    cursor = 0
    stem_vocabulary = tuple(args.stems)
    with torch.inference_mode():
        for batch in loader:
            if batch["contributions"] is None:
                raise ValueError("Convolutive SIR/SAR requires saved receiver-specific contributions")
            mixture = batch["mixture"]
            target = batch["target"]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = model(mixture.to(device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start
            estimate = output["estimate"].cpu().numpy()
            if model.config.mode == "girnet":
                predicted_graphs.append(output["adjacency"].cpu().numpy())
                target_graphs.append(batch["bleed_matrix"].numpy())
            mixture_np = mixture.numpy()
            target_np = target.numpy()
            components_np = batch["contributions"].numpy()
            for local_index in range(mixture.shape[0]):
                if "source_order_json" in batch:
                    source_order = tuple(json.loads(batch["source_order_json"][local_index]))
                else:
                    source_order = stem_vocabulary
                if len(source_order) != model.config.channels:
                    raise ValueError(
                        f"Expected {model.config.channels} ordered sources, got {source_order}"
                    )
                if not set(source_order).issubset(stem_vocabulary):
                    raise ValueError(f"Manifest source order {source_order} is outside {stem_vocabulary}")
                source_subset = (
                    batch["source_subset"][local_index]
                    if "source_subset" in batch else "+".join(source_order)
                )
                for channel, stem in enumerate(source_order):
                    reference_components = components_np[local_index, channel]
                    sisdr_in = si_sdr(target_np[local_index, channel], mixture_np[local_index, channel])
                    sisdr_out = si_sdr(target_np[local_index, channel], estimate[local_index, channel])
                    sir_in, _ = bss_projection_metrics(reference_components, mixture_np[local_index, channel], channel)
                    sir_out, sar_out = bss_projection_metrics(reference_components, estimate[local_index, channel], channel)
                    rows.append({
                        "index": cursor + local_index,
                        "track": batch["track"][local_index],
                        "start_sample": int(batch["start_sample"][local_index]),
                        "bleed_level_db": float(batch["bleed_level_db"][local_index]),
                        "rt60_seconds": float(batch["rt60_seconds"][local_index]),
                        "room_id": batch.get("room_id", [""] * mixture.shape[0])[local_index],
                        "graph_pattern_id": int(batch.get("graph_pattern_id", [-1] * mixture.shape[0])[local_index]),
                        "channel": channel,
                        "stem": stem,
                        "source_subset": source_subset,
                        "sisdr_in": sisdr_in,
                        "sisdr_out": sisdr_out,
                        "delta_sisdr": sisdr_out - sisdr_in,
                        "sir_in": sir_in,
                        "sir_out": sir_out,
                        "delta_sir": sir_out - sir_in,
                        "sar_out": sar_out,
                    })
            cursor += mixture.shape[0]
    with (args.output_dir / "metrics_per_stem.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = aggregate(rows)
    if predicted_graphs:
        target_array = np.concatenate(target_graphs)
        predicted_array = np.concatenate(predicted_graphs)
        summary["adjacency"] = adjacency_metrics(target_array, predicted_array)
        np.save(args.output_dir / "target_adjacency.npy", target_array)
        np.save(args.output_dir / "predicted_adjacency.npy", predicted_array)
    summary["inference"] = {
        "total_seconds": inference_seconds,
        "milliseconds_per_mixture": 1000.0 * inference_seconds / len(dataset),
        "real_time_factor": inference_seconds / (len(dataset) * 5.0),
        "device": str(device),
        "batch_size": args.batch_size,
    }
    save_json(args.output_dir / "metrics_summary.json", summary)
    bootstrap = track_cluster_bootstrap(rows, args.bootstrap_replicates, args.bootstrap_seed)
    save_json(args.output_dir / "track_bootstrap.json", bootstrap)
    write_summary_csv(args.output_dir / "track_bootstrap.csv", bootstrap)
    if args.reference_metrics_csv is not None:
        paired = paired_track_cluster_bootstrap(
            rows, read_metric_rows(args.reference_metrics_csv),
            args.bootstrap_replicates, args.bootstrap_seed,
        )
        save_json(args.output_dir / "paired_vs_reference.json", paired)
        write_summary_csv(args.output_dir / "paired_vs_reference.csv", paired)
    project = Path(__file__).resolve().parents[1]
    provenance = {
        "schema_version": 1,
        "evaluation_scope": args.evaluation_scope or (
            "convolutive pilot; descriptive only; not a final test set" if pre_rendered else
            "full convolutive evaluation with disjoint official MUSDB18HQ songs and room bank"
        ),
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": file_sha256(args.checkpoint)},
        "model": {"architecture_mode": model.config.mode,
                  "parameter_count": sum(parameter.numel() for parameter in model.parameters())},
        "configuration_snapshot": checkpoint.get("config"),
        "dataset": {"manifest": str(args.manifest.resolve()), "manifest_sha256": file_sha256(args.manifest),
                    "songs": len({row["track"] for row in rows}),
                    "mixtures": len(dataset), "source_estimates": len(rows),
                    "source_order": (
                        list(stem_vocabulary) if model.config.channels == len(stem_vocabulary)
                        else "entry-specific ordered subsets"
                    ),
                    "source_vocabulary": list(stem_vocabulary),
                    "channels": model.config.channels,
                    "sample_rate_hz": 44100, "segment_duration_seconds": 5.0,
                    "target": "diagonal RIR-filtered source"},
        "metric_definitions": {**METRIC_DEFINITIONS,
            "convolutive_sir_sar": (
                f"Ordered projection onto the {model.config.channels} receiver-specific h_ij*s_j components."
            )},
        "evaluation_script_sha256": file_sha256(Path(__file__)),
        "git_commit": git_commit(project),
        "seed": args.seed,
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
                      "unit": "complete song/track", "paired": args.reference_metrics_csv is not None},
        "software_versions": software_versions(),
    }
    save_json(args.output_dir / "provenance.json", provenance)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
