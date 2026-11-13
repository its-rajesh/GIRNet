#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from girnet.bootstrap import paired_track_cluster_bootstrap, read_metric_rows, track_cluster_bootstrap
from girnet.utils import save_json


EXPECTED = {
    "mean": 15.212512128372044,
    "ci95_low": 14.662915280819414,
    "ci95_high": 15.675007006015624,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify clustered-bootstrap regression against frozen GIRNet results")
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    rows = read_metric_rows(args.metrics_csv)
    summary = track_cluster_bootstrap(rows, args.replicates, args.seed)
    observed = summary["groups"]["overall"]["delta_sisdr"]
    checks = {
        name: math.isclose(float(observed[name]), expected, rel_tol=0.0, abs_tol=1e-12)
        for name, expected in EXPECTED.items()
    }
    paired_self = paired_track_cluster_bootstrap(rows, rows, 10_000, args.seed)
    paired_delta = paired_self["groups"]["overall"]["delta_sisdr"]
    checks["paired_self_zero"] = all(float(paired_delta[key]) == 0.0 for key in ("mean", "ci95_low", "ci95_high"))
    result = {
        "passed": all(checks.values()),
        "expected": EXPECTED,
        "observed": observed,
        "paired_self": paired_delta,
        "checks": checks,
        "metrics_csv": str(args.metrics_csv.resolve()),
        "replicates": args.replicates,
        "seed": args.seed,
    }
    save_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
