from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from girnet.convolutive import (
    ConvolutiveRenderDataset, calibrate_rirs, calibrate_rirs_from_levels, convolve_scene, pad_rirs, sample_geometry,
    simulate_raw_rirs, unpad_rirs,
)


class ConvolutiveTests(unittest.TestCase):
    def test_rir_calibration_matches_requested_db(self) -> None:
        rng = np.random.default_rng(1)
        raw = [[rng.normal(size=31 + i + j) for j in range(4)] for i in range(4)]
        effective, adjacency, levels, _ = calibrate_rirs(raw, -20.0, np.random.default_rng(2), 2.0)
        for i in range(4):
            self.assertEqual(adjacency[i, i], 0.0)
            for j in range(4):
                if i != j:
                    measured = 20.0 * np.log10(np.linalg.norm(effective[i][j]) / np.linalg.norm(effective[i][i]))
                    self.assertAlmostEqual(measured, levels[i, j], places=5)

    def test_independent_edge_level_calibration(self) -> None:
        rng = np.random.default_rng(12)
        raw = [[rng.normal(size=21 + i + j) for j in range(4)] for i in range(4)]
        levels = np.asarray([
            [0, -40, -35, -30], [-25, 0, -20, -15],
            [-10, -5, 0, 0], [-40, -20, -10, 0],
        ], dtype=np.float32)
        effective, adjacency, realized, _ = calibrate_rirs_from_levels(raw, levels)
        for i in range(4):
            self.assertEqual(adjacency[i, i], 0.0)
            for j in range(4):
                if i != j:
                    measured = 20.0 * np.log10(np.linalg.norm(effective[i][j]) / np.linalg.norm(effective[i][i]))
                    self.assertAlmostEqual(measured, levels[i, j], places=5)
                    self.assertAlmostEqual(realized[i, j], levels[i, j], places=5)

    def test_convolution_reconstructs_mixture(self) -> None:
        rng = np.random.default_rng(3)
        rirs = [[rng.normal(size=9) * 0.01 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            rirs[i][i][0] += 1.0
        sources = rng.normal(size=(4, 100 + 16)).astype(np.float32) * 0.02
        rendered = convolve_scene(sources, rirs, output_samples=100, context_samples=8)
        contributions = rendered["contributions"]
        np.testing.assert_allclose(rendered["mixture"], contributions.sum(axis=1), atol=2e-7, rtol=0)
        np.testing.assert_allclose(
            rendered["target"], np.stack([contributions[i, i] for i in range(4)]), atol=2e-7, rtol=0,
        )

    def test_geometry_is_deterministic_and_close_mics_are_close(self) -> None:
        first = sample_geometry(np.random.default_rng(44))
        second = sample_geometry(np.random.default_rng(44))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        distances = np.linalg.norm(first[1] - first[2], axis=1)
        self.assertTrue(np.all((distances >= 0.15) & (distances <= 0.30)))

    def test_direct_path_rirs_are_deterministic(self) -> None:
        dimensions, sources, microphones = sample_geometry(np.random.default_rng(55))
        first, _ = simulate_raw_rirs(dimensions, sources, microphones, 8000, 0.0)
        second, _ = simulate_raw_rirs(dimensions, sources, microphones, 8000, 0.0)
        for i in range(4):
            for j in range(4):
                np.testing.assert_array_equal(first[i][j], second[i][j])

    def test_pad_round_trip_and_manifest_renderer(self) -> None:
        rng = np.random.default_rng(7)
        raw = [[rng.normal(size=5 + i + j) * 0.01 for j in range(4)] for i in range(4)]
        for i in range(4):
            raw[i][i][0] += 1.0
        padded, lengths = pad_rirs(raw)
        restored = unpad_rirs(padded, lengths)
        for i in range(4):
            for j in range(4):
                np.testing.assert_allclose(restored[i][j], raw[i][j], rtol=1e-6)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = root / "track"
            track.mkdir()
            for stem_index, stem in enumerate(("vocals", "drums", "bass", "other")):
                signal = rng.normal(size=400).astype(np.float32) * (0.01 + stem_index * 0.001)
                sf.write(track / f"{stem}.wav", signal, 8000, subtype="FLOAT")
            rir_path = root / "room.npz"
            np.savez_compressed(rir_path, raw_rirs=padded, raw_rir_lengths=lengths)
            row = {
                "track_path": str(track), "start_sample": 100, "raw_rirs_path": str(rir_path),
                "nominal_bleed_db": -20.0, "edge_jitter_db": 2.0, "render_seed": 123,
                "requested_rt60_seconds": 0.2, "room_id": "room_test",
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = ConvolutiveRenderDataset(
                manifest, sample_rate=8000, segment_samples=100, include_contributions=True,
            )
            first, second = dataset[0], dataset[0]
            np.testing.assert_array_equal(first["mixture"].numpy(), second["mixture"].numpy())
            np.testing.assert_allclose(
                first["mixture"].numpy(), first["contributions"].numpy().sum(axis=1), atol=2e-7,
            )


if __name__ == "__main__":
    unittest.main()
