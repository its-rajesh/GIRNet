from __future__ import annotations

import unittest

from girnet.bootstrap import paired_track_cluster_bootstrap, track_cluster_bootstrap


def make_row(track: str, index: int, stem: str, value: float) -> dict:
    return {
        "index": index,
        "track": track,
        "start_sample": index * 100,
        "bleed_level_db": -20.0,
        "channel": 0 if stem == "vocals" else 1,
        "stem": stem,
        "sisdr_in": value - 1,
        "sisdr_out": value,
        "delta_sisdr": 1.0,
        "sir_in": value - 2,
        "sir_out": value,
        "delta_sir": 2.0,
        "sar_out": value + 1,
    }


class EvaluationReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = []
        for track_index, track in enumerate(("song_a", "song_b", "song_c")):
            for stem in ("vocals", "drums"):
                self.rows.append(make_row(track, track_index, stem, 10.0 + track_index))

    def test_complete_song_bootstrap_counts_and_constant_improvement(self) -> None:
        result = track_cluster_bootstrap(self.rows, replicates=10_000, seed=7)
        metric = result["groups"]["overall"]["delta_sisdr"]
        self.assertEqual(metric["independent_songs"], 3)
        self.assertEqual(metric["mixtures"], 3)
        self.assertEqual(metric["source_estimates"], 6)
        self.assertEqual(metric["mean"], 1.0)
        self.assertEqual(metric["ci95_low"], 1.0)
        self.assertEqual(metric["ci95_high"], 1.0)

    def test_paired_bootstrap_preserves_pairing(self) -> None:
        candidate = [{**row, "sisdr_out": float(row["sisdr_out"]) + 0.5} for row in self.rows]
        result = paired_track_cluster_bootstrap(candidate, self.rows, replicates=10_000, seed=9)
        metric = result["groups"]["overall"]["sisdr_out"]
        self.assertAlmostEqual(metric["mean"], 0.5)
        self.assertAlmostEqual(metric["ci95_low"], 0.5)
        self.assertAlmostEqual(metric["ci95_high"], 0.5)

    def test_heterogeneous_graph_patterns_are_distinct_pairing_units(self) -> None:
        base = make_row("song_a", 0, "vocals", 10.0)
        rows = [
            {**base, "rt60_seconds": 0.2, "room_id": "room_a", "graph_pattern_id": 0},
            {**base, "rt60_seconds": 0.2, "room_id": "room_a", "graph_pattern_id": 1,
             "sisdr_out": 11.0},
        ]
        result = track_cluster_bootstrap(rows, replicates=10_000, seed=5)
        self.assertEqual(result["groups"]["overall"]["sisdr_out"]["mixtures"], 2)
        candidate = [{**row, "sisdr_out": float(row["sisdr_out"]) + 0.25} for row in rows]
        paired = paired_track_cluster_bootstrap(candidate, rows, replicates=10_000, seed=5)
        self.assertAlmostEqual(paired["groups"]["overall"]["sisdr_out"]["mean"], 0.25)

    def test_convolutive_conditions_are_distinct_pairing_units(self) -> None:
        base = make_row("song_a", 0, "vocals", 10.0)
        rows = [
            {**base, "rt60_seconds": 0.2, "room_id": "room_a"},
            {**base, "rt60_seconds": 0.4, "room_id": "room_b", "sisdr_out": 11.0},
        ]
        result = track_cluster_bootstrap(rows, replicates=10_000, seed=5)
        self.assertEqual(result["groups"]["overall"]["sisdr_out"]["mixtures"], 2)
        candidate = [{**row, "sisdr_out": float(row["sisdr_out"]) + 0.25} for row in rows]
        paired = paired_track_cluster_bootstrap(candidate, rows, replicates=10_000, seed=5)
        self.assertAlmostEqual(paired["groups"]["overall"]["sisdr_out"]["mean"], 0.25)


if __name__ == "__main__":
    unittest.main()
