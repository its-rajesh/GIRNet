#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from girnet.utils import save_json


LEVELS = (-40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0)
STEMS = ("vocals", "drums", "bass", "other")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def edge_offsets(seed: int) -> np.ndarray:
    # All nine levels occur within every mixture; three levels occur twice.
    values = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 3, 6], dtype=int)
    values = np.random.default_rng(seed).permutation(values)
    matrix = np.zeros((4, 4), dtype=int)
    matrix[~np.eye(4, dtype=bool)] = values
    return matrix


def matrix_for_pattern(pattern: int, offsets: np.ndarray) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.float32)
    mask = ~np.eye(4, dtype=bool)
    indices = (offsets[mask] + int(pattern)) % len(LEVELS)
    matrix[mask] = np.asarray(LEVELS, dtype=np.float32)[indices]
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced heterogeneous directed-edge manifests")
    parser.add_argument("--source-dir", type=Path, default=Path("paper_spl/convolutive/full_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_spl/convolutive/heterogeneous_v1"))
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets = edge_offsets(args.seed)
    level_index = {value: index for index, value in enumerate(LEVELS)}
    summary = {
        "protocol_version": "heterogeneous_directed_edges_v1",
        "status": "manifest_only_no_training_started",
        "source_protocol": str(args.source_dir.resolve()),
        "seed": args.seed,
        "levels_db": list(LEVELS),
        "rt60_seconds": [0.0, 0.2, 0.4, 0.6, 0.8],
        "source_order": list(STEMS),
        "edge_offset_indices": offsets.tolist(),
        "design": (
            "Each mixture contains all nine bleed levels across its 12 directed off-diagonal edges. "
            "Across nine patterns, every directed edge takes every level exactly once."
        ),
        "splits": {},
    }
    split_tracks = {}
    split_rooms = {}
    for split in ("train", "val", "test"):
        source_path = args.source_dir / f"{split}.jsonl"
        source_rows = read_jsonl(source_path)
        rows = []
        edge_counts = {(i, j): Counter() for i in range(4) for j in range(4) if i != j}
        for row in source_rows:
            original_level = float(row["nominal_bleed_db"])
            if original_level not in level_index:
                raise ValueError(f"Unexpected source level: {original_level}")
            pattern = level_index[original_level]
            levels = matrix_for_pattern(pattern, offsets)
            mask = ~np.eye(4, dtype=bool)
            updated = dict(row)
            updated.update({
                "example_id": f"hetero_{row['example_id']}",
                "heterogeneous_graph": True,
                "graph_pattern_id": pattern,
                "edge_levels_db": levels.tolist(),
                "nominal_bleed_db": float(levels[mask].mean()),
                "original_uniform_nominal_bleed_db": original_level,
                "edge_jitter_db": 0.0,
                "heterogeneous_target_definition": (
                    "Twelve independently specified directed off-diagonal acoustic-path levels; "
                    "all nine levels occur within each mixture."
                ),
            })
            rows.append(updated)
            for i in range(4):
                for j in range(4):
                    if i != j:
                        edge_counts[(i, j)][float(levels[i, j])] += 1
        output_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(output_path, rows)
        split_tracks[split] = {row["track_path"] for row in rows}
        split_rooms[split] = {row["room_id"] for row in rows}
        all_counts = [count for counter in edge_counts.values() for count in counter.values()]
        summary["splits"][split] = {
            "source_manifest": str(source_path.resolve()),
            "source_manifest_sha256": file_sha256(source_path),
            "manifest": str(output_path.resolve()),
            "manifest_sha256": file_sha256(output_path),
            "examples": len(rows),
            "independent_tracks": len(split_tracks[split]),
            "rooms": len(split_rooms[split]),
            "edge_level_count_min": min(all_counts),
            "edge_level_count_max": max(all_counts),
            "unique_levels_per_mixture_min": min(
                len(set(np.asarray(row["edge_levels_db"])[~np.eye(4, dtype=bool)].tolist())) for row in rows
            ),
        }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    summary["invariants"] = {
        "track_splits_pairwise_disjoint": all(not (split_tracks[a] & split_tracks[b]) for a, b in pairs),
        "room_splits_pairwise_disjoint": all(not (split_rooms[a] & split_rooms[b]) for a, b in pairs),
        "all_nine_levels_in_every_mixture": all(
            summary["splits"][split]["unique_levels_per_mixture_min"] == 9
            for split in ("train", "val", "test")
        ),
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
