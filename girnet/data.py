from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset


DEFAULT_STEMS = ("vocals", "drums", "bass", "other")


@dataclass(frozen=True)
class ManifestEntry:
    track: str
    split: str
    source_split: str
    start_sample: int
    source_sample_rate: int
    segment_samples: int
    nominal_bleed_db: float | None
    seed: int
    edge_jitter_db: float = 0.0

    @classmethod
    def from_dict(cls, item: dict) -> "ManifestEntry":
        return cls(**item)


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(ManifestEntry.from_dict(json.loads(line)))
    if not entries:
        raise ValueError(f"Manifest is empty: {path}")
    return entries


def write_manifest(path: str | Path, entries: Iterable[ManifestEntry]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.__dict__, sort_keys=True) + "\n")


def _read_audio_segment(path: Path, start: int, frames: int) -> tuple[np.ndarray, int]:
    with sf.SoundFile(path) as audio:
        sample_rate = int(audio.samplerate)
        audio.seek(min(start, len(audio)))
        signal = audio.read(frames=frames, dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if len(signal) < frames:
        signal = np.pad(signal, (0, frames - len(signal)))
    return signal.astype(np.float32, copy=False), sample_rate


def _resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(signal, target_rate // divisor, source_rate // divisor).astype(np.float32)


def make_bleed_matrix(
    channels: int,
    nominal_db: float,
    rng: np.random.Generator,
    edge_jitter_db: float = 0.0,
) -> np.ndarray:
    """Return A where A[i,j] is source j's amplitude in microphone i."""
    db = np.full((channels, channels), float(nominal_db), dtype=np.float32)
    if edge_jitter_db > 0.0:
        db += rng.uniform(-edge_jitter_db, edge_jitter_db, size=db.shape).astype(np.float32)
    db = np.minimum(db, 0.0)
    matrix = np.power(10.0, db / 20.0).astype(np.float32)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def mix_stems(stems: np.ndarray, bleed_matrix: np.ndarray, peak: float = 0.98) -> tuple[np.ndarray, np.ndarray]:
    if stems.ndim != 2 or bleed_matrix.shape != (stems.shape[0], stems.shape[0]):
        raise ValueError(f"Bad shapes: stems={stems.shape}, A={bleed_matrix.shape}")
    mixture = stems + bleed_matrix @ stems
    maximum = max(float(np.max(np.abs(mixture))), float(np.max(np.abs(stems))), 1e-8)
    gain = min(1.0, float(peak) / maximum)
    return (mixture * gain).astype(np.float32), (stems * gain).astype(np.float32)


class MUSDBBleedDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        stems: list[str] | tuple[str, ...] = DEFAULT_STEMS,
        sample_rate: int = 22050,
        segment_samples: int = 110250,
        train_bleed_levels_db: list[float] | tuple[float, ...] = (-40, -30, -20, -10, 0),
        train_edge_jitter_db: float = 2.0,
        peak: float = 0.98,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.entries = read_manifest(self.manifest_path)
        self.stems = tuple(stems)
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(segment_samples)
        self.train_levels = tuple(float(x) for x in train_bleed_levels_db)
        self.train_edge_jitter_db = float(train_edge_jitter_db)
        self.peak = float(peak)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.entries)

    def _level_and_rng(self, index: int, entry: ManifestEntry) -> tuple[float, np.random.Generator, float]:
        dynamic = entry.nominal_bleed_db is None
        seed = int(entry.seed + index * 1009 + (self.epoch * 1_000_003 if dynamic else 0))
        rng = np.random.default_rng(seed)
        level = float(rng.choice(self.train_levels)) if dynamic else float(entry.nominal_bleed_db)
        jitter = self.train_edge_jitter_db if dynamic else float(entry.edge_jitter_db)
        return level, rng, jitter

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        entry = self.entries[index]
        track_dir = Path(entry.track)
        source_frames = int(round(self.segment_samples * entry.source_sample_rate / self.sample_rate))
        signals = []
        for stem in self.stems:
            signal, rate = _read_audio_segment(track_dir / f"{stem}.wav", entry.start_sample, source_frames)
            if rate != entry.source_sample_rate:
                raise ValueError(f"Manifest/audio rate mismatch for {track_dir}: {entry.source_sample_rate} vs {rate}")
            signal = _resample(signal, rate, self.sample_rate)
            if len(signal) < self.segment_samples:
                signal = np.pad(signal, (0, self.segment_samples - len(signal)))
            signals.append(signal[: self.segment_samples])
        clean = np.stack(signals).astype(np.float32)
        level, rng, jitter = self._level_and_rng(index, entry)
        matrix = make_bleed_matrix(len(self.stems), level, rng, jitter)
        mixture, clean = mix_stems(clean, matrix, self.peak)
        return {
            "mixture": torch.from_numpy(mixture),
            "target": torch.from_numpy(clean),
            "bleed_matrix": torch.from_numpy(matrix),
            "bleed_level_db": level,
            "track": track_dir.name,
            "start_sample": entry.start_sample,
            "index": index,
        }
