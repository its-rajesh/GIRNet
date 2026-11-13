from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from girnet.data import MUSDBBleedDataset, ManifestEntry, make_bleed_matrix, mix_stems, write_manifest
from girnet.losses import GIRNetLoss, LossWeights, ordered_si_sdr
from girnet.model import GIRNetSeparator, ModelConfig
from scripts.build_channel_count_benchmark import LEVELS, edge_level_matrix, source_subsets


class GIRNetSmokeTests(unittest.TestCase):
    def test_channel_count_protocol_and_dynamic_graph_shapes(self):
        for channels, subset_count in ((2, 6), (3, 4), (4, 1)):
            self.assertEqual(len(source_subsets(channels)), subset_count)
            seen = {
                (receiver, sender): set()
                for receiver in range(channels) for sender in range(channels)
                if receiver != sender
            }
            for pattern in range(len(LEVELS)):
                matrix = edge_level_matrix(channels, pattern)
                self.assertEqual(matrix.shape, (channels, channels))
                self.assertTrue(np.allclose(np.diag(matrix), 0.0))
                if channels < 4:
                    self.assertEqual(
                        len(set(matrix[~np.eye(channels, dtype=bool)].tolist())),
                        channels * (channels - 1),
                    )
                for receiver, sender in seen:
                    seen[(receiver, sender)].add(float(matrix[receiver, sender]))
            self.assertTrue(all(values == set(LEVELS) for values in seen.values()))

            model = GIRNetSeparator(ModelConfig(
                mode="girnet", channels=channels, base_channels=4, depth=2,
                kernel_size=5, bottleneck_blocks=1, attention_heads=1, edge_hidden=8,
            ))
            mixture = torch.randn(2, channels, 257)
            output = model(mixture)
            self.assertEqual(output["estimate"].shape, mixture.shape)
            self.assertEqual(output["adjacency"].shape, (2, channels, channels))
            self.assertTrue(torch.allclose(
                output["adjacency"].diagonal(dim1=-2, dim2=-1),
                torch.zeros(2, channels),
            ))

    def test_bleed_matrix_and_mixture_convention(self):
        rng = np.random.default_rng(3)
        matrix = make_bleed_matrix(3, -20.0, rng, 0.0)
        self.assertTrue(np.allclose(np.diag(matrix), 0.0))
        self.assertTrue(np.allclose(matrix[~np.eye(3, dtype=bool)], 0.1))
        stems = np.eye(3, dtype=np.float32)
        mixture, target = mix_stems(stems, matrix, peak=10.0)
        self.assertTrue(np.allclose(target, stems))
        self.assertTrue(np.allclose(mixture, stems + matrix @ stems))

    def test_all_internal_variants_forward(self):
        mixture = torch.randn(2, 4, 257)
        for mode in ("local", "tac", "attention", "girnet"):
            config = ModelConfig(mode=mode, channels=4, base_channels=4, depth=2, kernel_size=5, bottleneck_blocks=2, attention_heads=2, edge_hidden=8)
            model = GIRNetSeparator(config)
            output = model(mixture)
            self.assertEqual(output["estimate"].shape, mixture.shape)
            self.assertEqual(output["adjacency"].shape, (2, 4, 4))
            self.assertTrue(torch.allclose(output["adjacency"].diagonal(dim1=-2, dim2=-1), torch.zeros(2, 4)))

    def test_graph_override_is_explicit_and_preserves_prediction(self):
        config = ModelConfig(
            mode="girnet", channels=4, base_channels=4, depth=2, kernel_size=5,
            bottleneck_blocks=1, attention_heads=1, edge_hidden=8,
        )
        model = GIRNetSeparator(config).eval()
        mixture = torch.randn(2, 4, 257)
        with torch.no_grad():
            default = model(mixture)
            override = torch.zeros(2, 4, 4)
            intervened = model(mixture, adjacency_override=override)
        torch.testing.assert_close(default["adjacency"], intervened["adjacency"])
        torch.testing.assert_close(intervened["used_adjacency"], override)
        self.assertFalse(torch.equal(default["estimate"], intervened["estimate"]))
        local = GIRNetSeparator(ModelConfig(
            mode="local", channels=4, base_channels=4, depth=2, kernel_size=5,
            bottleneck_blocks=1, attention_heads=1, edge_hidden=8,
        ))
        with self.assertRaises(ValueError):
            local(mixture, adjacency_override=override)

    def test_graph_loss_backward_without_pit(self):
        config = ModelConfig(mode="girnet", channels=3, base_channels=4, depth=2, kernel_size=5, bottleneck_blocks=1, attention_heads=1, edge_hidden=8)
        model = GIRNetSeparator(config)
        mixture = torch.randn(2, 3, 256)
        target = torch.randn(2, 3, 256)
        matrix = torch.from_numpy(np.stack([make_bleed_matrix(3, -20.0, np.random.default_rng(i)) for i in range(2)]))
        criterion = GIRNetLoss(
            LossWeights(fft_sizes=(32, 64), hsic_max_frames=16),
            graph_mode=True,
        )
        loss, terms = criterion(model(mixture), target, matrix)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("adjacency", terms)

    def test_sensitive_losses_promote_half_inputs_to_float32(self):
        target = torch.randn(2, 3, 256).half() * 1e-4
        estimate = target + torch.randn_like(target) * 1e-5
        score = ordered_si_sdr(target, estimate)
        self.assertEqual(score.dtype, torch.float32)
        self.assertTrue(torch.isfinite(score).all())

    def test_identity_loss_only_uses_low_bleed_examples(self):
        config = ModelConfig(mode="girnet", channels=3, base_channels=4, depth=2, kernel_size=5, bottleneck_blocks=1, attention_heads=1, edge_hidden=8)
        model = GIRNetSeparator(config)
        mixture = torch.randn(2, 3, 256)
        target = mixture.clone()
        matrix = torch.from_numpy(np.stack([make_bleed_matrix(3, -40.0, np.random.default_rng(i)) for i in range(2)]))
        criterion = GIRNetLoss(LossWeights(fft_sizes=(32,), identity_weight=1.0), graph_mode=True)
        _, terms = criterion(
            model(mixture), target, matrix, mixture=mixture,
            bleed_level_db=torch.tensor([-40.0, -10.0]),
        )
        self.assertTrue(torch.isfinite(terms["identity"]))
        self.assertGreaterEqual(float(terms["identity"].detach()), 0.0)

    def test_manifest_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = root / "track"
            track.mkdir()
            rate, samples = 8000, 800
            time = np.arange(samples) / rate
            stems = ("vocals", "drums", "bass", "other")
            for index, stem in enumerate(stems):
                sf.write(track / f"{stem}.wav", 0.1 * np.sin(2 * np.pi * (100 + 20 * index) * time), rate)
            manifest = root / "test.jsonl"
            write_manifest(manifest, [ManifestEntry(
                track=str(track), split="test", source_split="test", start_sample=0,
                source_sample_rate=rate, segment_samples=samples, nominal_bleed_db=-20.0, seed=1,
                edge_jitter_db=0.0,
            )])
            dataset = MUSDBBleedDataset(manifest, stems=stems, sample_rate=rate, segment_samples=samples)
            item = dataset[0]
            self.assertEqual(tuple(item["mixture"].shape), (4, samples))
            self.assertEqual(tuple(item["target"].shape), (4, samples))
            self.assertEqual(tuple(item["bleed_matrix"].shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
