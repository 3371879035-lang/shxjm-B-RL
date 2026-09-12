"""Replay ISR-v1 against either the frozen c88abfa module tree or current code."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-root", required=True,
                        help="directory that directly contains the brl package")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="160000:160050,260000:260200")
    args = parser.parse_args()
    module_root = Path(args.module_root).resolve()
    sys.path.insert(0, str(module_root))

    from brl.independent_candidate import IndependentCandidate
    from brl.local_env import RadioEnv, Source

    def sources_for(seed: int, mode: int, kind: str):
        rng = np.random.default_rng(seed)
        count = int(rng.integers(10, 17))
        sources = []
        for channel in rng.choice(np.arange(1, 21), count, replace=False):
            if kind == "edge":
                radius_from_origin = rng.uniform(1650, 1800)
                receive_radius = 1000.0
            else:
                radius_from_origin = 1800.0 * np.sqrt(rng.random())
                receive_radius = rng.uniform(1000, 1500)
            angle = rng.uniform(0, 2 * np.pi)
            position = radius_from_origin * np.array([np.cos(angle), np.sin(angle)])
            source_kind, direction = "omni", 0.0
            if mode == 4 and (kind == "edge" or rng.random() < 0.5):
                source_kind = "directional"
                direction = float(np.degrees(angle)) if kind == "edge" else float(rng.uniform(0, 360))
            sources.append(Source(int(channel), position, float(receive_radius),
                                  source_kind, direction))
        return sources

    seed_cases = []
    for part in args.seeds.split(","):
        start, stop = (int(value) for value in part.split(":"))
        seed_cases.extend((seed, "edge" if offset % 3 == 0 else "uniform")
                          for offset, seed in enumerate(range(start, stop)))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    csv_path = out / "per_case.csv"
    for mode in (3, 4):
        for seed, kind in seed_cases:
            sources = sources_for(seed, mode, kind)
            env = RadioEnv(mode=mode, n_sources=len(sources), seed=seed,
                           step_limit=20000, bearing_decimals=2)
            env.reset(seed=seed, sources=sources)
            result = IndependentCandidate(mode=mode).run(env)
            reconstructed = (env.move_distance / 5 + 5 * env.n_measure + env.n_switch
                             + 3 * env.n_clear_fail + 5 * env.cleared_count())
            row = {
                "mode": mode, "seed": seed, "kind": kind,
                "sources": len(sources), "success": bool(result["success"]),
                "cleared": int(env.cleared_count()),
                "virtual_time_s": float(env.virtual_time),
                "time_per_source_s": float(env.virtual_time / len(sources)),
                "distance_m": float(env.move_distance), "measures": int(env.n_measure),
                "switches": int(env.n_switch), "clear_calls": int(env.n_clear),
                "failed_clear": int(env.n_clear_fail),
                "cost_identity_error_s": float(env.virtual_time - reconstructed),
            }
            rows.append(row)
            with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader(); writer.writerows(rows)
            if not row["success"] or row["cleared"] != row["sources"]:
                raise RuntimeError(f"ISR baseline failed: {row}")
    summary = {"module_root": str(module_root), "cases": len(rows), "modes": {}}
    for mode in (3, 4):
        group = [row for row in rows if row["mode"] == mode]
        summary["modes"][str(mode)] = {
            "cases": len(group),
            "mean_time_per_source_s": float(np.mean([r["time_per_source_s"] for r in group])),
            "max_cost_identity_error_s": float(max(abs(r["cost_identity_error_s"]) for r in group)),
        }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
