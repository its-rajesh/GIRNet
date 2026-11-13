#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from girnet.convolutive import pad_rirs, sample_geometry, simulate_raw_rirs
from girnet.data import read_manifest
from girnet.utils import save_json


STEMS = ("vocals", "drums", "bass", "other")


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_crops(entries: list) -> list:
    seen: set[tuple[str, int]] = set()
    result = []
    for entry in entries:
        key = (entry.track, entry.start_sample)
        if key not in seen:
            seen.add(key)
            result.append(entry)
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_rooms(
    output: Path, split: str, count_per_rt60: int, rt60_values: list[float], sample_rate: int,
    seed: int,
) -> list[dict]:
    rooms: list[dict] = []
    split_code = {"train": 11, "val": 23, "test": 37}[split]
    room_dir = output / "rooms" / split
    room_dir.mkdir(parents=True)
    for rt_index, rt60 in enumerate(rt60_values):
        for room_index in range(count_per_rt60):
            room_seed = seed + split_code * 10_000_019 + rt_index * 100_003 + room_index * 1_009
            rng = np.random.default_rng(room_seed)
            dimensions, sources, microphones = sample_geometry(rng, channels=len(STEMS))
            raw_rirs, acoustic = simulate_raw_rirs(
                dimensions, sources, microphones, sample_rate, rt60,
            )
            padded, lengths = pad_rirs(raw_rirs)
            room_id = f"{split}_rt{rt_index:02d}_room{room_index:02d}"
            rir_path = room_dir / f"{room_id}.npz"
            metadata_path = room_dir / f"{room_id}.json"
            np.savez_compressed(rir_path, raw_rirs=padded, raw_rir_lengths=lengths)
            metadata = {
                "room_id": room_id,
                "room_split": split,
                "seed": room_seed,
                "room_dimensions_m": dimensions.tolist(),
                "source_positions_m": sources.tolist(),
                "microphone_positions_m": microphones.tolist(),
                "close_mic_distances_m": np.linalg.norm(sources - microphones, axis=1).tolist(),
                "rir_context_samples": int(lengths.max()) - 1,
                "raw_rirs_path": str(rir_path.resolve()),
                **acoustic,
            }
            save_json(metadata_path, metadata)
            rooms.append(metadata)
    return rooms


def context_safe_crops(entries: list, context: int, segment_samples: int, sample_rate: int) -> tuple[list, list, list]:
    accepted, rejected, adjusted = [], [], []
    accepted_keys: set[tuple[str, int]] = set()
    for entry in unique_crops(entries):
        audio = sf.info(Path(entry.track) / "vocals.wav")
        if int(audio.samplerate) != sample_rate:
            rejected.append({"track": entry.track, "start_sample": entry.start_sample, "reason": "sample_rate"})
            continue
        maximum_start = int(audio.frames) - segment_samples - context
        if maximum_start < context:
            rejected.append({"track": entry.track, "start_sample": entry.start_sample, "reason": "track_too_short"})
            continue
        safe_start = int(np.clip(entry.start_sample, context, maximum_start))
        key = (entry.track, safe_start)
        if key in accepted_keys:
            rejected.append({"track": entry.track, "start_sample": entry.start_sample, "reason": "duplicate_after_shift"})
            continue
        accepted_keys.add(key)
        accepted.append(replace(entry, start_sample=safe_start))
        if safe_start != entry.start_sample:
            adjusted.append({
                "track": entry.track, "original_start_sample": entry.start_sample,
                "adjusted_start_sample": safe_start,
            })
    return accepted, rejected, adjusted


def make_row(entry, split: str, room: dict, level: float, edge_jitter_db: float, seed: int,
             sample_rate: int, segment_samples: int) -> dict:
    identity = f"{split}|{entry.track}|{entry.start_sample}|{room['room_id']}|{level:g}"
    render_seed = seed + int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % 2_000_000_000
    return {
        "example_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
        "split": split,
        "source_split": entry.source_split,
        "track": Path(entry.track).name,
        "track_path": str(Path(entry.track).resolve()),
        "start_sample": int(entry.start_sample),
        "sample_rate": sample_rate,
        "segment_samples": segment_samples,
        "room_id": room["room_id"],
        "room_split": room["room_split"],
        "raw_rirs_path": room["raw_rirs_path"],
        "rir_context_samples": int(room["rir_context_samples"]),
        "requested_rt60_seconds": float(room["requested_rt60_seconds"]),
        "measured_mean_rt60_seconds": float(room["measured_mean_rt60_seconds"]),
        "nominal_bleed_db": float(level),
        "edge_jitter_db": float(edge_jitter_db),
        "render_seed": int(render_seed),
        "target_definition": "diagonal RIR-filtered source h_ii*s_i",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build disjoint, balanced GIRNet convolutive room banks and manifests")
    parser.add_argument("--instantaneous-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--levels-db", type=parse_floats, default=parse_floats("-40,-35,-30,-25,-20,-15,-10,-5,0"))
    parser.add_argument("--rt60-seconds", type=parse_floats, default=parse_floats("0,0.2,0.4,0.6,0.8"))
    parser.add_argument("--train-rooms-per-rt60", type=int, default=5)
    parser.add_argument("--val-rooms-per-rt60", type=int, default=2)
    parser.add_argument("--test-rooms-per-rt60", type=int, default=2)
    parser.add_argument("--edge-jitter-db", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_samples = int(round(args.segment_seconds * args.sample_rate))
    conditions = list(product(args.rt60_seconds, args.levels_db))
    room_counts = {
        "train": args.train_rooms_per_rt60, "val": args.val_rooms_per_rt60,
        "test": args.test_rooms_per_rt60,
    }
    rooms = {
        split: build_rooms(
            args.output_dir, split, room_counts[split], args.rt60_seconds, args.sample_rate, args.seed,
        )
        for split in ("train", "val", "test")
    }

    summary: dict = {
        "status": "manifest_only_no_model_training_started",
        "protocol_version": "convolutive_full_v2_boundary_safe",
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "segment_samples": segment_samples,
        "segment_duration_seconds": args.segment_seconds,
        "stems": list(STEMS),
        "levels_db": args.levels_db,
        "rt60_seconds": args.rt60_seconds,
        "edge_jitter_db": args.edge_jitter_db,
        "target_definition": "diagonal RIR-filtered source h_ii*s_i; no dereverberation",
        "splits": {},
    }
    all_tracks: dict[str, set[str]] = {}
    all_room_ids: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        source_manifest = args.instantaneous_dir / f"{split}.jsonl"
        source_entries = read_manifest(source_manifest)
        maximum_context = max(int(room["rir_context_samples"]) for room in rooms[split])
        crops, rejected, adjusted = context_safe_crops(
            source_entries, maximum_context, segment_samples, args.sample_rate,
        )
        if not crops:
            raise RuntimeError(f"No context-safe crops for {split}")
        rng = np.random.default_rng(args.seed + {"train": 101, "val": 202, "test": 303}[split])
        crop_order = rng.permutation(len(crops)).tolist()
        rows: list[dict] = []
        if split == "train":
            for sequence, crop_index in enumerate(crop_order):
                rt60, level = conditions[sequence % len(conditions)]
                candidates = [room for room in rooms[split] if room["requested_rt60_seconds"] == rt60]
                room = candidates[(sequence // len(conditions)) % len(candidates)]
                rows.append(make_row(crops[crop_index], split, room, level, args.edge_jitter_db,
                                     args.seed, args.sample_rate, segment_samples))
        else:
            for crop_rank, crop_index in enumerate(crop_order):
                for condition_index, (rt60, level) in enumerate(conditions):
                    candidates = [room for room in rooms[split] if room["requested_rt60_seconds"] == rt60]
                    room = candidates[(crop_rank + condition_index) % len(candidates)]
                    rows.append(make_row(crops[crop_index], split, room, level, args.edge_jitter_db,
                                         args.seed, args.sample_rate, segment_samples))
        manifest_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(manifest_path, rows)
        all_tracks[split] = {row["track_path"] for row in rows}
        all_room_ids[split] = {row["room_id"] for row in rows}
        condition_counts = Counter((row["requested_rt60_seconds"], row["nominal_bleed_db"]) for row in rows)
        summary["splits"][split] = {
            "source_manifest": str(source_manifest.resolve()),
            "source_manifest_sha256": sha256(source_manifest),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "examples": len(rows),
            "independent_tracks": len(all_tracks[split]),
            "context_safe_crops": len(crops),
            "boundary_adjusted_crops": len(adjusted),
            "rejected_crops": len(rejected),
            "rejection_reasons": dict(Counter(item["reason"] for item in rejected)),
            "maximum_rir_context_samples": maximum_context,
            "rooms": len(all_room_ids[split]),
            "condition_count_min": min(condition_counts.values()),
            "condition_count_max": max(condition_counts.values()),
        }
        save_json(
            args.output_dir / f"{split}_crop_adjustments.json",
            {"adjusted": adjusted, "rejected": rejected},
        )

    summary["invariants"] = {
        "track_splits_pairwise_disjoint": all(
            not (all_tracks[a] & all_tracks[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        ),
        "room_splits_pairwise_disjoint": all(
            not (all_room_ids[a] & all_room_ids[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        ),
        "all_eval_conditions_present": all(
            summary["splits"][split]["condition_count_min"] > 0 for split in ("val", "test")
        ),
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
