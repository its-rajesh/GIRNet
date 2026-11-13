from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

try:
    import pyroomacoustics as pra
except ImportError:  # Baseline environments only need the saved-RIR renderer.
    pra = None


def sample_geometry(
    rng: np.random.Generator,
    channels: int = 4,
    room_min: tuple[float, float, float] = (5.0, 4.0, 2.7),
    room_max: tuple[float, float, float] = (8.0, 7.0, 3.5),
    wall_margin: float = 0.7,
    min_source_distance: float = 0.9,
    close_mic_range: tuple[float, float] = (0.15, 0.30),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    room = rng.uniform(room_min, room_max).astype(np.float64)
    sources: list[np.ndarray] = []
    for _ in range(channels):
        for _attempt in range(10_000):
            candidate = rng.uniform(np.full(3, wall_margin), room - wall_margin)
            if all(np.linalg.norm(candidate - source) >= min_source_distance for source in sources):
                sources.append(candidate)
                break
        else:
            raise RuntimeError("Could not sample separated source positions")
    microphones = []
    for source in sources:
        for _attempt in range(1_000):
            direction = rng.normal(size=3)
            direction[2] *= 0.35
            direction /= np.linalg.norm(direction)
            distance = rng.uniform(*close_mic_range)
            microphone = source + distance * direction
            if np.all(microphone >= 0.1) and np.all(microphone <= room - 0.1):
                microphones.append(microphone)
                break
        else:
            raise RuntimeError("Could not place a close microphone")
    return room, np.stack(sources), np.stack(microphones)


def simulate_raw_rirs(
    room_dimensions: np.ndarray,
    source_positions: np.ndarray,
    microphone_positions: np.ndarray,
    sample_rate: int,
    rt60_seconds: float,
    rt60_relative_tolerance: float = 0.08,
    rt60_calibration_attempts: int = 4,
) -> tuple[list[list[np.ndarray]], dict]:
    if pra is None:
        raise ImportError("pyroomacoustics is required to create rooms, but not to render saved RIR manifests")
    if rt60_seconds <= 0.0:
        design_rt60, absorption, max_order, attempts = 0.0, 1.0, 0, 1
    else:
        design_rt60 = float(rt60_seconds)
    for attempt in range(1, max(1, rt60_calibration_attempts) + 1):
        if rt60_seconds > 0.0:
            absorption, max_order = pra.inverse_sabine(design_rt60, room_dimensions.tolist())
        room = pra.ShoeBox(
            room_dimensions.tolist(), fs=int(sample_rate), materials=pra.Material(absorption),
            max_order=int(max_order), air_absorption=rt60_seconds > 0.0,
        )
        for source in source_positions:
            room.add_source(source.tolist())
        for microphone in microphone_positions:
            room.add_microphone(microphone.tolist())
        room.compute_rir()
        rirs = [
            [np.asarray(room.rir[microphone][source], dtype=np.float64) for source in range(len(source_positions))]
            for microphone in range(len(microphone_positions))
        ]
        if rt60_seconds <= 0.0:
            measured = [0.0] * len(source_positions)
            attempts = attempt
            break
        measured = [
            float(pra.experimental.rt60.measure_rt60(rirs[index][index], fs=sample_rate))
            for index in range(len(source_positions))
        ]
        attempts = attempt
        measured_mean = float(np.mean(measured))
        relative_error = abs(measured_mean - rt60_seconds) / rt60_seconds
        if relative_error <= rt60_relative_tolerance:
            break
        design_rt60 = float(np.clip(design_rt60 * rt60_seconds / max(measured_mean, 1e-3), 0.05, 2.0))
    return rirs, {
        "requested_rt60_seconds": float(rt60_seconds),
        "measured_diagonal_rt60_seconds": measured,
        "measured_mean_rt60_seconds": float(np.mean(measured)),
        "sabine_design_rt60_seconds": float(design_rt60),
        "rt60_relative_error": 0.0 if rt60_seconds <= 0 else abs(float(np.mean(measured)) - rt60_seconds) / rt60_seconds,
        "rt60_calibration_attempts": attempts,
        "sabine_absorption": float(absorption),
        "image_source_max_order": int(max_order),
    }


def calibrate_rirs(
    raw_rirs: list[list[np.ndarray]],
    nominal_bleed_db: float,
    rng: np.random.Generator,
    edge_jitter_db: float = 0.0,
) -> tuple[list[list[np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    channels = len(raw_rirs)
    effective: list[list[np.ndarray]] = [[None for _ in range(channels)] for _ in range(channels)]  # type: ignore[list-item]
    gains = np.ones((channels, channels), dtype=np.float64)
    levels = np.zeros((channels, channels), dtype=np.float64)
    adjacency = np.zeros((channels, channels), dtype=np.float64)
    for receiver in range(channels):
        diagonal_norm = np.linalg.norm(raw_rirs[receiver][receiver])
        if diagonal_norm <= 0:
            raise ValueError(f"Zero-energy diagonal RIR for microphone {receiver}")
        for source in range(channels):
            raw = np.asarray(raw_rirs[receiver][source], dtype=np.float64)
            if receiver == source:
                effective[receiver][source] = raw.copy()
                continue
            jitter = rng.uniform(-edge_jitter_db, edge_jitter_db) if edge_jitter_db > 0 else 0.0
            level = min(0.0, float(nominal_bleed_db) + float(jitter))
            desired_ratio = 10.0 ** (level / 20.0)
            raw_norm = np.linalg.norm(raw)
            if raw_norm <= 0:
                raise ValueError(f"Zero-energy off-diagonal RIR ({receiver},{source})")
            gain = desired_ratio * diagonal_norm / raw_norm
            effective[receiver][source] = raw * gain
            gains[receiver, source] = gain
            levels[receiver, source] = level
            adjacency[receiver, source] = desired_ratio
    return effective, adjacency.astype(np.float32), levels.astype(np.float32), gains.astype(np.float32)


def calibrate_rirs_from_levels(
    raw_rirs: list[list[np.ndarray]],
    edge_levels_db: np.ndarray | list[list[float]],
) -> tuple[list[list[np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate every directed off-diagonal RIR to its independently specified level."""
    channels = len(raw_rirs)
    levels = np.asarray(edge_levels_db, dtype=np.float64)
    if levels.shape != (channels, channels):
        raise ValueError(f"Expected edge-level matrix {(channels, channels)}, got {levels.shape}")
    if np.any(levels[~np.eye(channels, dtype=bool)] > 0.0):
        raise ValueError("Off-diagonal bleed levels must not exceed 0 dB")
    levels = levels.copy()
    np.fill_diagonal(levels, 0.0)
    effective: list[list[np.ndarray]] = [[None for _ in range(channels)] for _ in range(channels)]  # type: ignore[list-item]
    gains = np.ones((channels, channels), dtype=np.float64)
    adjacency = np.zeros((channels, channels), dtype=np.float64)
    for receiver in range(channels):
        diagonal_norm = np.linalg.norm(raw_rirs[receiver][receiver])
        if diagonal_norm <= 0:
            raise ValueError(f"Zero-energy diagonal RIR for microphone {receiver}")
        for source_index in range(channels):
            raw = np.asarray(raw_rirs[receiver][source_index], dtype=np.float64)
            if receiver == source_index:
                effective[receiver][source_index] = raw.copy()
                continue
            raw_norm = np.linalg.norm(raw)
            if raw_norm <= 0:
                raise ValueError(f"Zero-energy off-diagonal RIR ({receiver},{source_index})")
            desired_ratio = 10.0 ** (levels[receiver, source_index] / 20.0)
            gain = desired_ratio * diagonal_norm / raw_norm
            effective[receiver][source_index] = raw * gain
            gains[receiver, source_index] = gain
            adjacency[receiver, source_index] = desired_ratio
    return effective, adjacency.astype(np.float32), levels.astype(np.float32), gains.astype(np.float32)


def pad_rirs(rirs: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    channels = len(rirs)
    lengths = np.asarray([[len(rirs[i][j]) for j in range(channels)] for i in range(channels)], dtype=np.int32)
    padded = np.zeros((channels, channels, int(lengths.max())), dtype=np.float32)
    for i in range(channels):
        for j in range(channels):
            padded[i, j, : lengths[i, j]] = rirs[i][j]
    return padded, lengths


def unpad_rirs(padded: np.ndarray, lengths: np.ndarray) -> list[list[np.ndarray]]:
    if padded.ndim != 3 or lengths.ndim != 2 or padded.shape[:2] != lengths.shape:
        raise ValueError(f"Incompatible RIR shapes: padded={padded.shape}, lengths={lengths.shape}")
    return [
        [np.asarray(padded[i, j, : int(lengths[i, j])], dtype=np.float64) for j in range(padded.shape[1])]
        for i in range(padded.shape[0])
    ]


def convolve_scene(
    extended_sources: np.ndarray,
    effective_rirs: list[list[np.ndarray]],
    output_samples: int,
    context_samples: int,
    peak: float = 0.98,
) -> dict[str, np.ndarray | float]:
    channels = extended_sources.shape[0]
    required = output_samples + 2 * context_samples
    if extended_sources.shape != (channels, required):
        raise ValueError(f"Expected source shape ({channels},{required}), got {extended_sources.shape}")
    contributions = np.zeros((channels, channels, output_samples), dtype=np.float64)
    for receiver in range(channels):
        for source in range(channels):
            full = fftconvolve(extended_sources[source].astype(np.float64), effective_rirs[receiver][source], mode="full")
            contributions[receiver, source] = full[context_samples: context_samples + output_samples]
    target = np.stack([contributions[i, i] for i in range(channels)])
    mixture = contributions.sum(axis=1)
    dry = extended_sources[:, context_samples: context_samples + output_samples].astype(np.float64)
    maximum = max(float(np.max(np.abs(mixture))), float(np.max(np.abs(target))), 1e-12)
    normalization_gain = min(1.0, float(peak) / maximum)
    mixture *= normalization_gain
    target *= normalization_gain
    contributions *= normalization_gain
    return {
        "mixture": mixture.astype(np.float32),
        "target": target.astype(np.float32),
        "dry": dry.astype(np.float32),
        "contributions": contributions.astype(np.float32),
        "normalization_gain": normalization_gain,
    }


def signal_level_graph(contributions: np.ndarray, floor_db: float = -100.0) -> np.ndarray:
    channels = contributions.shape[0]
    result = np.zeros((channels, channels), dtype=np.float32)
    for receiver in range(channels):
        diagonal = np.linalg.norm(contributions[receiver, receiver].astype(np.float64))
        for source in range(channels):
            if receiver != source:
                ratio = np.linalg.norm(contributions[receiver, source].astype(np.float64)) / max(diagonal, 1e-12)
                result[receiver, source] = max(floor_db, 20.0 * math.log10(max(ratio, 1e-12)))
    return result


def read_extended_sources(
    track: str | Path,
    stems: list[str] | tuple[str, ...],
    center_start_sample: int,
    output_samples: int,
    context_samples: int,
    sample_rate: int,
) -> np.ndarray:
    start = center_start_sample - context_samples
    frames = output_samples + 2 * context_samples
    if start < 0:
        raise ValueError("Selected source crop lacks left RIR context")
    signals = []
    for stem in stems:
        path = Path(track) / f"{stem}.wav"
        with sf.SoundFile(path) as audio:
            if int(audio.samplerate) != int(sample_rate):
                raise ValueError(f"Unexpected sample rate for {path}: {audio.samplerate}")
            if start + frames > len(audio):
                raise ValueError("Selected source crop lacks right RIR context")
            audio.seek(start)
            signal = audio.read(frames, dtype="float32", always_2d=True).mean(axis=1)
        signals.append(signal)
    return np.stack(signals).astype(np.float32)


class ConvolutiveManifestDataset(Dataset):
    def __init__(self, manifest: str | Path) -> None:
        self.manifest_path = Path(manifest)
        with self.manifest_path.open(encoding="utf-8") as handle:
            self.entries = [json.loads(line) for line in handle if line.strip()]
        if not self.entries:
            raise ValueError(f"Empty convolutive manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        audio = np.load(entry["audio_path"])
        return {
            "mixture": torch.from_numpy(audio["mixture"].astype(np.float32)),
            "target": torch.from_numpy(audio["target"].astype(np.float32)),
            "dry": torch.from_numpy(audio["dry"].astype(np.float32)),
            "contributions": torch.from_numpy(audio["contributions"].astype(np.float32)) if "contributions" in audio else None,
            "bleed_matrix": torch.tensor(entry["acoustic_path_adjacency"], dtype=torch.float32),
            "bleed_level_db": float(entry["nominal_bleed_db"]),
            "rt60_seconds": float(entry["requested_rt60_seconds"]),
            "track": entry["track"],
            "start_sample": int(entry["start_sample"]),
            "index": index,
        }


class ConvolutiveRenderDataset(Dataset):
    """Render deterministic convolutive examples from compact manifests and raw RIR banks."""

    def __init__(
        self,
        manifest: str | Path,
        stems: list[str] | tuple[str, ...] = ("vocals", "drums", "bass", "other"),
        sample_rate: int = 44100,
        segment_samples: int = 220500,
        peak: float = 0.98,
        include_contributions: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest)
        with self.manifest_path.open(encoding="utf-8") as handle:
            self.entries = [json.loads(line) for line in handle if line.strip()]
        if not self.entries:
            raise ValueError(f"Empty convolutive manifest: {manifest}")
        self.stems = tuple(stems)
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(segment_samples)
        self.peak = float(peak)
        self.include_contributions = bool(include_contributions)
        self._rir_cache: dict[str, tuple[list[list[np.ndarray]], int]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def set_epoch(self, epoch: int) -> None:
        # Full-v2 rows deliberately define fixed examples. Keeping this method
        # makes the dataset interchangeable with the instantaneous trainer.
        self.epoch = int(epoch)

    def _load_rirs(self, path: str) -> tuple[list[list[np.ndarray]], int]:
        if path not in self._rir_cache:
            with np.load(path) as archive:
                raw = unpad_rirs(archive["raw_rirs"], archive["raw_rir_lengths"])
                context = int(np.max(archive["raw_rir_lengths"])) - 1
            self._rir_cache[path] = raw, context
        return self._rir_cache[path]

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        raw_rirs, context = self._load_rirs(entry["raw_rirs_path"])
        source_indices = tuple(int(value) for value in entry.get("source_indices", range(len(raw_rirs))))
        if len(set(source_indices)) != len(source_indices):
            raise ValueError(f"Duplicate source_indices in manifest row {index}: {source_indices}")
        if any(value < 0 or value >= len(raw_rirs) for value in source_indices):
            raise ValueError(f"Out-of-range source_indices in manifest row {index}: {source_indices}")
        raw_rirs = [[raw_rirs[i][j] for j in source_indices] for i in source_indices]
        source_order = tuple(entry.get("source_order", self.stems))
        if len(source_order) != len(source_indices):
            raise ValueError(
                f"source_order/source_indices mismatch in row {index}: "
                f"{source_order} versus {source_indices}"
            )
        if "edge_levels_db" in entry:
            effective, adjacency, actual_levels, _ = calibrate_rirs_from_levels(
                raw_rirs, entry["edge_levels_db"],
            )
        else:
            effective, adjacency, actual_levels, _ = calibrate_rirs(
                raw_rirs,
                float(entry["nominal_bleed_db"]),
                np.random.default_rng(int(entry["render_seed"])),
                float(entry["edge_jitter_db"]),
            )
        extended = read_extended_sources(
            entry["track_path"], source_order, int(entry["start_sample"]), self.segment_samples,
            context, self.sample_rate,
        )
        rendered = convolve_scene(extended, effective, self.segment_samples, context, self.peak)
        result = {
            "mixture": torch.from_numpy(rendered["mixture"]),
            "target": torch.from_numpy(rendered["target"]),
            "bleed_matrix": torch.from_numpy(adjacency),
            "bleed_level_db": float(entry["nominal_bleed_db"]),
            "realized_acoustic_path_db": torch.from_numpy(actual_levels),
            "heterogeneous_graph": bool(entry.get("heterogeneous_graph", False)),
            "graph_pattern_id": int(entry.get("graph_pattern_id", -1)),
            "source_order_json": json.dumps(source_order),
            "source_subset": str(entry.get("source_subset", "+".join(source_order))),
            "rt60_seconds": float(entry["requested_rt60_seconds"]),
            "room_id": entry["room_id"],
            "track": Path(entry["track_path"]).name,
            "start_sample": int(entry["start_sample"]),
            "normalization_gain": float(rendered["normalization_gain"]),
            "index": index,
        }
        if self.include_contributions:
            result["contributions"] = torch.from_numpy(rendered["contributions"])
            result["dry"] = torch.from_numpy(rendered["dry"])
            result["signal_level_bleed_db"] = torch.from_numpy(signal_level_graph(rendered["contributions"]))
        return result
