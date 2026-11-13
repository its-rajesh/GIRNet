#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from girnet.data import DEFAULT_STEMS, ManifestEntry, write_manifest
from girnet.utils import save_json


def parse_levels(value: str) -> list[float]:
    levels = [float(item) for item in value.split(",")]
    if not levels or min(levels) < -80.0 or max(levels) > 0.0:
        raise argparse.ArgumentTypeError("Bleed levels must be comma-separated dB values in [-80, 0]")
    return levels


def track_directories(root: Path, source_split: str) -> list[Path]:
    directories = sorted(path for path in (root / source_split).iterdir() if path.is_dir())
    if not directories:
        raise FileNotFoundError(f"No tracks found under {root / source_split}")
    return directories


def valid_starts(
    track: Path,
    stems: tuple[str, ...],
    seconds: float,
    hop_seconds: float,
    min_stem_rms_db: float,
) -> tuple[list[int], int]:
    info = sf.info(track / f"{stems[0]}.wav")
    rate = int(info.samplerate)
    frames = int(round(seconds * rate))
    hop = int(round(hop_seconds * rate))
    candidates = list(range(0, max(1, info.frames - frames + 1), max(1, hop)))
    if not candidates:
        candidates = [0]
    valid = []
    threshold = 10.0 ** (min_stem_rms_db / 20.0)
    handles = [sf.SoundFile(track / f"{stem}.wav") for stem in stems]
    try:
        for start in candidates:
            active = True
            for handle in handles:
                handle.seek(min(start, len(handle)))
                signal = handle.read(frames=frames, dtype="float32", always_2d=True).mean(axis=1)
                rms = float(np.sqrt(np.mean(signal * signal) + 1e-12))
                if rms < threshold:
                    active = False
                    break
            if active:
                valid.append(start)
    finally:
        for handle in handles:
            handle.close()
    return valid, rate


def entries_for_tracks(
    tracks: list[Path],
    split: str,
    source_split: str,
    levels: list[float] | None,
    segment_seconds: float,
    target_sample_rate: int,
    hop_seconds: float,
    min_stem_rms_db: float,
    stems: tuple[str, ...],
    seed: int,
    max_clips_per_track: int | None,
    eval_edge_jitter_db: float,
) -> list[ManifestEntry]:
    entries = []
    target_samples = int(round(segment_seconds * target_sample_rate))
    for track_index, track in enumerate(tracks):
        starts, source_rate = valid_starts(track, stems, segment_seconds, hop_seconds, min_stem_rms_db)
        if max_clips_per_track is not None and len(starts) > max_clips_per_track:
            positions = np.linspace(0, len(starts) - 1, max_clips_per_track).round().astype(int)
            starts = [starts[position] for position in positions]
        for clip_index, start in enumerate(starts):
            assigned_levels = levels if levels is not None else [None]
            for level_index, level in enumerate(assigned_levels):
                entry_seed = seed + track_index * 100_003 + clip_index * 1_009 + level_index
                entries.append(ManifestEntry(
                    track=str(track.resolve()),
                    split=split,
                    source_split=source_split,
                    start_sample=int(start),
                    source_sample_rate=source_rate,
                    segment_samples=target_samples,
                    nominal_bleed_db=level,
                    seed=int(entry_seed),
                    edge_jitter_db=float(eval_edge_jitter_db if level is not None else 0.0),
                ))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe MUSDB18HQ GIRNet manifests")
    parser.add_argument("--musdb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", default=list(DEFAULT_STEMS))
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--hop-seconds", type=float, default=5.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--levels-db", type=parse_levels, default=parse_levels("-40,-35,-30,-25,-20,-15,-10,-5,0"))
    parser.add_argument("--min-stem-rms-db", type=float, default=-50.0)
    parser.add_argument("--eval-edge-jitter-db", type=float, default=2.0)
    parser.add_argument("--val-clips-per-track", type=int, default=2)
    parser.add_argument("--test-clips-per-track", type=int, default=2)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    train_pool = track_directories(args.musdb_root, "train")
    test_tracks = track_directories(args.musdb_root, "test")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_pool))
    val_count = max(1, int(round(len(train_pool) * args.val_fraction)))
    val_tracks = sorted(train_pool[index] for index in order[:val_count])
    train_tracks = sorted(train_pool[index] for index in order[val_count:])
    stems = tuple(args.stems)

    common = dict(
        segment_seconds=args.segment_seconds,
        target_sample_rate=args.sample_rate,
        hop_seconds=args.hop_seconds,
        min_stem_rms_db=args.min_stem_rms_db,
        stems=stems,
        seed=args.seed,
        eval_edge_jitter_db=args.eval_edge_jitter_db,
    )
    train_entries = entries_for_tracks(train_tracks, "train", "train", None, max_clips_per_track=None, **common)
    val_entries = entries_for_tracks(val_tracks, "val", "train", args.levels_db, max_clips_per_track=args.val_clips_per_track, **common)
    test_entries = entries_for_tracks(test_tracks, "test", "test", args.levels_db, max_clips_per_track=args.test_clips_per_track, **common)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "train.jsonl", train_entries)
    write_manifest(args.output_dir / "val.jsonl", val_entries)
    write_manifest(args.output_dir / "test.jsonl", test_entries)
    represented = {
        "train": sorted({Path(entry.track).name for entry in train_entries}),
        "val": sorted({Path(entry.track).name for entry in val_entries}),
        "test": sorted({Path(entry.track).name for entry in test_entries}),
    }
    summary = {
        "musdb_root": str(args.musdb_root.resolve()),
        "seed": args.seed,
        "stems": list(stems),
        "sample_rate": args.sample_rate,
        "segment_seconds": args.segment_seconds,
        "levels_db": args.levels_db,
        "eval_edge_jitter_db": args.eval_edge_jitter_db,
        "train_tracks": [path.name for path in train_tracks],
        "val_tracks": [path.name for path in val_tracks],
        "test_tracks": [path.name for path in test_tracks],
        "entries": {"train": len(train_entries), "val": len(val_entries), "test": len(test_entries)},
        "represented_tracks": {name: len(items) for name, items in represented.items()},
        "skipped_tracks_without_active_chunks": {
            "train": sorted(set(path.name for path in train_tracks) - set(represented["train"])),
            "val": sorted(set(path.name for path in val_tracks) - set(represented["val"])),
            "test": sorted(set(path.name for path in test_tracks) - set(represented["test"])),
        },
        "split_invariants": {
            "train_val_track_overlap": bool(set(train_tracks) & set(val_tracks)),
            "official_test_used_for_training": False,
        },
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["entries"], indent=2))


if __name__ == "__main__":
    main()
