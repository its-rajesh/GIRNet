#!/usr/bin/env python3
"""Build a common rotating-edge-level protocol for separate C=2,3,4 runs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def source_subsets(channels: int) -> list[tuple[int, ...]]:
    if channels < 2 or channels > len(STEMS):
        raise ValueError(f"Unsupported ordered channel count: {channels}")
    return list(itertools.combinations(range(len(STEMS)), channels))


def edge_level_matrix(channels: int, pattern: int) -> np.ndarray:
    """Apply one rotating rule for every C; edge positions rotate through LEVELS."""
    edge_count = channels * (channels - 1)
    offsets = np.arange(edge_count, dtype=int) % len(LEVELS)
    values = np.asarray(LEVELS, dtype=np.float32)[(offsets + int(pattern)) % len(LEVELS)]
    matrix = np.zeros((channels, channels), dtype=np.float32)
    matrix[~np.eye(channels, dtype=bool)] = values
    return matrix


def build_split(source_rows: list[dict], channels: int) -> tuple[list[dict], dict]:
    subsets = source_subsets(channels)
    level_index = {level: index for index, level in enumerate(LEVELS)}
    track_phase = {
        track: index for index, track in enumerate(sorted({str(row["track_path"]) for row in source_rows}))
    }
    counters = Counter()
    output: list[dict] = []
    coverage = {
        (subset_index, receiver, sender): Counter()
        for subset_index in range(len(subsets))
        for receiver in range(channels)
        for sender in range(channels)
        if receiver != sender
    }
    subset_examples = Counter()
    for row in source_rows:
        original_level = float(row["nominal_bleed_db"])
        if original_level not in level_index:
            raise ValueError(f"Unexpected source level {original_level}")
        pattern = level_index[original_level]
        # Rotate within each song and pattern, with a stable song-specific
        # phase. Test songs have enough examples for every pair/triplet, so
        # subset-specific song-cluster intervals retain all independent songs.
        track_key = str(row["track_path"])
        counter_key = (track_key, pattern)
        phase = track_phase[track_key]
        subset_index = (phase + counters[counter_key]) % len(subsets)
        counters[counter_key] += 1
        indices = subsets[subset_index]
        names = tuple(STEMS[index] for index in indices)
        levels = edge_level_matrix(channels, pattern)
        off_diagonal = levels[~np.eye(channels, dtype=bool)]
        updated = dict(row)
        updated.update({
            "example_id": f"channel_c{channels}_{'+'.join(names)}_{row['example_id']}",
            "channel_count_protocol": "ordered_rotating_levels_v1",
            "channels": channels,
            "source_indices": list(indices),
            "source_order": list(names),
            "source_subset": "+".join(names),
            "source_subset_index": subset_index,
            "heterogeneous_graph": True,
            "graph_pattern_id": pattern,
            "edge_levels_db": levels.tolist(),
            # Trainer compatibility: this is explicitly the strongest directed
            # edge, not a claimed single physical level for the whole mixture.
            "nominal_bleed_db": float(off_diagonal.max()),
            "strongest_edge_level_db": float(off_diagonal.max()),
            "mean_edge_level_db": float(off_diagonal.mean()),
            "original_uniform_nominal_bleed_db": original_level,
            "edge_jitter_db": 0.0,
            "channel_count_target_definition": (
                "Distinct rotating directed-edge levels where possible; every local directed "
                "edge takes every level across the split."
            ),
        })
        output.append(updated)
        subset_examples[subset_index] += 1
        for receiver in range(channels):
            for sender in range(channels):
                if receiver != sender:
                    coverage[(subset_index, receiver, sender)][float(levels[receiver, sender])] += 1

    missing = {
        str(key): sorted(set(LEVELS) - set(counter))
        for key, counter in coverage.items() if set(counter) != set(LEVELS)
    }
    if missing:
        raise RuntimeError(f"Incomplete per-subset edge-level coverage: {missing}")
    counts = [value for counter in coverage.values() for value in counter.values()]
    summary = {
        "examples": len(output),
        "subsets": ["+".join(STEMS[index] for index in subset) for subset in subsets],
        "examples_by_subset": {
            "+".join(STEMS[index] for index in subsets[key]): value
            for key, value in sorted(subset_examples.items())
        },
        "directed_edges_per_example": channels * (channels - 1),
        "edge_level_count_min": min(counts),
        "edge_level_count_max": max(counts),
        "every_subset_local_edge_sees_all_levels": True,
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("paper_spl/convolutive/full_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_spl/channel_count/manifests"))
    parser.add_argument("--channels", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    requested = sorted(set(args.channels))
    if requested != [2, 3, 4]:
        raise ValueError("The paper benchmark must be generated jointly for C=2,3,4")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    split_tracks: dict[str, set[str]] = {}
    split_rooms: dict[str, set[str]] = {}
    summary = {
        "protocol_version": "ordered_channel_count_rotating_levels_v1",
        "status": "validated_manifests_only_training_not_started",
        "source_protocol": str(args.source_dir.resolve()),
        "levels_db": list(LEVELS),
        "source_vocabulary": list(STEMS),
        "segment_duration_seconds": 5.0,
        "sample_rate_hz": 44100,
        "assignment": (
            "For E=C(C-1) row-major directed edges, offsets 0..E-1 modulo nine are "
            "cyclically shifted by graph_pattern_id. Source subsets are round-robin balanced "
            "within each pattern."
        ),
        "channel_counts": {},
    }
    for channels in requested:
        channel_summary = {"directed_edges": channels * (channels - 1), "splits": {}}
        for split in ("train", "val", "test"):
            source_path = args.source_dir / f"{split}.jsonl"
            source_rows = read_jsonl(source_path)
            rows, split_summary = build_split(source_rows, channels)
            output_path = args.output_dir / f"c{channels}" / f"{split}.jsonl"
            write_jsonl(output_path, rows)
            split_tracks.setdefault(split, {row["track_path"] for row in rows})
            split_rooms.setdefault(split, {row["room_id"] for row in rows})
            if split_tracks[split] != {row["track_path"] for row in rows}:
                raise RuntimeError(f"C={channels} changed the {split} song split")
            if split_rooms[split] != {row["room_id"] for row in rows}:
                raise RuntimeError(f"C={channels} changed the {split} room split")
            channel_summary["splits"][split] = {
                **split_summary,
                "manifest": str(output_path.resolve()),
                "manifest_sha256": file_sha256(output_path),
                "source_manifest": str(source_path.resolve()),
                "source_manifest_sha256": file_sha256(source_path),
                "independent_tracks": len(split_tracks[split]),
                "rooms": len(split_rooms[split]),
            }
        summary["channel_counts"][f"c{channels}"] = channel_summary

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    summary["invariants"] = {
        "track_splits_pairwise_disjoint": all(not (split_tracks[a] & split_tracks[b]) for a, b in pairs),
        "room_splits_pairwise_disjoint": all(not (split_rooms[a] & split_rooms[b]) for a, b in pairs),
        "same_example_count_across_channel_counts": all(
            len(read_jsonl(args.output_dir / f"c{channels}" / f"{split}.jsonl"))
            == len(read_jsonl(args.source_dir / f"{split}.jsonl"))
            for channels in requested for split in ("train", "val", "test")
        ),
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
