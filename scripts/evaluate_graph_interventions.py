#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from girnet.convolutive import ConvolutiveRenderDataset
from girnet.losses import ordered_si_sdr
from girnet.metrics import adjacency_metrics
from girnet.model import GIRNetSeparator, ModelConfig
from girnet.provenance import file_sha256, git_commit, software_versions
from girnet.utils import save_json, seed_everything


STEMS = ("vocals", "drums", "bass", "other")
MODES = ("predicted", "oracle", "shuffled_oracle", "uniform_oracle", "zero")


def derangement(size: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    for _ in range(10_000):
        permutation = rng.permutation(size)
        if np.all(permutation != identity):
            return torch.from_numpy(permutation.astype(np.int64))
    raise RuntimeError("Could not construct edge derangement")


def interventions(oracle: torch.Tensor, permutation: torch.Tensor) -> dict[str, torch.Tensor | None]:
    batch, channels, _ = oracle.shape
    mask = ~torch.eye(channels, device=oracle.device, dtype=torch.bool)
    values = oracle[:, mask]
    shuffled = torch.zeros_like(oracle)
    shuffled[:, mask] = values[:, permutation.to(values.device)]
    uniform = torch.zeros_like(oracle)
    uniform[:, mask] = values.mean(dim=1, keepdim=True).expand(-1, values.shape[1])
    return {
        "predicted": None,
        "oracle": oracle,
        "shuffled_oracle": shuffled,
        "uniform_oracle": uniform,
        "zero": torch.zeros_like(oracle),
    }


def clustered_summary(rows: list[dict], field: str, draws: np.ndarray,
                      source: str | None = None, reference_field: str | None = None) -> dict:
    selected = [row for row in rows if source is None or row["stem"] == source]
    tracks = sorted({row["track"] for row in rows})
    index = {track: position for position, track in enumerate(tracks)}
    sums = np.zeros(len(tracks), dtype=np.float64)
    counts = np.zeros(len(tracks), dtype=np.int64)
    values = []
    for row in selected:
        value = float(row[field])
        if reference_field is not None:
            value -= float(row[reference_field])
        position = index[row["track"]]
        sums[position] += value
        counts[position] += 1
        values.append(value)
    if np.any(counts == 0):
        raise ValueError("Every song must be represented in every intervention group")
    replicates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(replicates, [2.5, 97.5])
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_std": float(replicates.std(ddof=1)),
        "independent_songs": len(tracks),
        "mixtures": len({(row["track"], row["start_sample"], row["rt60_seconds"], row["room_id"],
                           row["graph_pattern_id"]) for row in selected}),
        "source_estimates": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal GIRNet graph intervention evaluation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1337)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.bootstrap_replicates < 10_000:
        raise ValueError("Use at least 10,000 song-cluster bootstrap replicates")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ConvolutiveRenderDataset(args.manifest, include_contributions=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GIRNetSeparator(ModelConfig(**checkpoint["model_config"])).to(device)
    if model.config.mode != "girnet":
        raise ValueError("Graph intervention evaluation requires a GIRNet checkpoint")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    permutation = derangement(12, args.seed + 91)
    rows = []
    predicted_graphs, target_graphs = [], []
    inference_seconds = {mode: 0.0 for mode in MODES}
    cursor = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            mixture = batch["mixture"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            oracle = batch["bleed_matrix"].to(device, non_blocking=True)
            input_scores = ordered_si_sdr(target, mixture).cpu().numpy()
            estimates = {}
            predicted_graph = None
            for mode, override in interventions(oracle, permutation).items():
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                output = model(mixture, adjacency_override=override)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds[mode] += time.perf_counter() - started
                estimates[mode] = ordered_si_sdr(target, output["estimate"]).cpu().numpy()
                if mode == "predicted":
                    predicted_graph = output["adjacency"].cpu().numpy()
                del output
            predicted_graphs.append(predicted_graph)
            target_graphs.append(oracle.cpu().numpy())
            count = mixture.shape[0]
            for local_index in range(count):
                for channel, stem in enumerate(STEMS):
                    row = {
                        "index": cursor + local_index,
                        "track": batch["track"][local_index],
                        "start_sample": int(batch["start_sample"][local_index]),
                        "rt60_seconds": float(batch["rt60_seconds"][local_index]),
                        "room_id": batch["room_id"][local_index],
                        "graph_pattern_id": int(batch["graph_pattern_id"][local_index]),
                        "channel": channel,
                        "stem": stem,
                        "sisdr_in": float(input_scores[local_index, channel]),
                    }
                    for mode in MODES:
                        value = float(estimates[mode][local_index, channel])
                        row[f"sisdr_{mode}"] = value
                        row[f"sisdr_improvement_{mode}"] = value - row["sisdr_in"]
                    rows.append(row)
            cursor += count
            if args.progress_every > 0 and (batch_index + 1) % args.progress_every == 0:
                print(json.dumps({"mixtures_complete": cursor, "mixtures_total": len(dataset)}), flush=True)
    metrics_path = args.output_dir / "metrics_per_source.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    track_names = sorted({row["track"] for row in rows})
    rng = np.random.default_rng(args.bootstrap_seed)
    draws = rng.integers(0, len(track_names), size=(args.bootstrap_replicates, len(track_names)))
    summary = {
        "method": "complete-song percentile cluster bootstrap",
        "replicates": args.bootstrap_replicates,
        "seed": args.bootstrap_seed,
        "interventions": {},
        "paired_vs_predicted": {},
        "by_source": {},
        "adjacency": adjacency_metrics(np.concatenate(target_graphs), np.concatenate(predicted_graphs)),
        "inference": {},
    }
    for mode in MODES:
        summary["interventions"][mode] = {
            "sisdr_out": clustered_summary(rows, f"sisdr_{mode}", draws),
            "sisdr_improvement": clustered_summary(rows, f"sisdr_improvement_{mode}", draws),
        }
        summary["inference"][mode] = {
            "total_seconds": inference_seconds[mode],
            "milliseconds_per_mixture": 1000.0 * inference_seconds[mode] / len(dataset),
        }
        summary["by_source"][mode] = {
            stem: clustered_summary(rows, f"sisdr_improvement_{mode}", draws, source=stem)
            for stem in STEMS
        }
        if mode != "predicted":
            summary["paired_vs_predicted"][mode] = clustered_summary(
                rows, f"sisdr_{mode}", draws, reference_field="sisdr_predicted",
            )
    save_json(args.output_dir / "intervention_summary.json", summary)
    np.save(args.output_dir / "predicted_adjacency.npy", np.concatenate(predicted_graphs))
    provenance = {
        "schema_version": 1,
        "experiment": "causal graph intervention on heterogeneous directed-edge mixtures",
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": file_sha256(args.checkpoint)},
        "configuration_snapshot": checkpoint["config"],
        "model": {"mode": model.config.mode, "parameters": sum(p.numel() for p in model.parameters())},
        "dataset": {"manifest": str(args.manifest.resolve()), "sha256": file_sha256(args.manifest),
                    "songs": len(track_names), "mixtures": len(dataset), "source_estimates": len(rows)},
        "intervention_definitions": {
            "predicted": "Unmodified learned predicted graph.",
            "oracle": "Ground-truth heterogeneous directed acoustic-path graph.",
            "shuffled_oracle": "Oracle edge-value multiset assigned through a fixed derangement of the 12 directed edges.",
            "uniform_oracle": "All off-diagonal edges equal the per-mixture mean oracle amplitude.",
            "zero": "All graph messages suppressed with a zero adjacency matrix.",
        },
        "bootstrap": {"unit": "complete song", "replicates": args.bootstrap_replicates,
                      "seed": args.bootstrap_seed, "confidence_level": 0.95},
        "random_seed": args.seed,
        "evaluation_script_sha256": file_sha256(Path(__file__)),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "software_versions": software_versions(),
    }
    save_json(args.output_dir / "provenance.json", provenance)
    lines = [
        "# Causal graph intervention results", "",
        "All values are SI-SDR improvement in dB with 95% complete-song cluster confidence intervals.", "",
        "| Graph used for message passing | SI-SDRi | Difference vs predicted |", "|---|---:|---:|",
    ]
    for mode in MODES:
        value = summary["interventions"][mode]["sisdr_improvement"]
        paired = summary["paired_vs_predicted"].get(mode)
        difference = "reference" if paired is None else f'{paired["mean"]:.3f} [{paired["ci95_low"]:.3f}, {paired["ci95_high"]:.3f}]'
        lines.append(
            f'| {mode.replace("_", " ")} | {value["mean"]:.3f} '
            f'[{value["ci95_low"]:.3f}, {value["ci95_high"]:.3f}] | {difference} |'
        )
    lines += ["", "Interpretation rule: oracle > predicted indicates graph-estimation headroom; predicted/oracle > shuffled, uniform, or zero demonstrates that edge identity and graph-mediated communication causally affect separation."]
    (args.output_dir / "CAUSAL_INTERVENTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["interventions"], indent=2))


if __name__ == "__main__":
    main()
