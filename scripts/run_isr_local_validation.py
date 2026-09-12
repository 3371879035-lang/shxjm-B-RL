"""Frozen local comparison for protocol-aligned G25OR and ISR.

This is an offline evaluation harness.  Source truth is used only here to build
paired scenarios and verify completion; neither policy receives it directly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from brl.g25o import G25OPolicy
from brl.independent_candidate import IndependentCandidate
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources


class EndpointErrorField:
    def value(self, channel: int, x: float, y: float) -> float:
        return 1.0 if np.sin(0.01 * x + 0.013 * y + channel) > 0 else -1.0


class ConstantErrorField:
    def value(self, channel: int, x: float, y: float) -> float:
        return 1.0


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(mode: int, seed: int, kind: str, variant: str, error_mode: str) -> dict:
    sources = generate_sources(seed, mode, kind)
    env = RadioEnv(
        mode=mode, n_sources=len(sources), seed=seed,
        step_limit=20000, bearing_decimals=2,
    )
    env.reset(seed=seed, n_sources=len(sources), sources=sources)
    if error_mode == "endpoints":
        env.error_field = EndpointErrorField()
    elif error_mode == "constant":
        env.error_field = ConstantErrorField()
    elif error_mode != "spatial":
        raise ValueError(f"unknown error mode: {error_mode}")

    started = time.perf_counter()
    if variant == "G25OR":
        result = G25OPolicy("G25OR", coverage="S25").run(env)
    elif variant == "ISR":
        result = IndependentCandidate(mode).run(env)
    else:
        raise ValueError(variant)
    wall_s = time.perf_counter() - started

    source_count = len(sources)
    cleared = env.cleared_count()
    reconstructed = (
        env.move_distance / 5.0 + 5.0 * env.n_measure + env.n_switch
        + 3.0 * env.n_clear_fail + 5.0 * cleared
    )
    success = bool(result.get("success") and env.completion_certificate())
    return {
        "mode": mode, "seed": seed, "kind": kind, "error_mode": error_mode,
        "variant": variant, "sources": source_count, "cleared": cleared,
        "success": success, "virtual_time_s": float(env.virtual_time),
        "seconds_per_source": float(env.virtual_time / source_count),
        "distance_m": float(env.move_distance), "measure_calls": int(env.n_measure),
        "switches": int(env.n_switch), "clear_calls": int(env.n_clear),
        "failed_clear": int(env.n_clear_fail),
        "solver_failures": int(result.get("solver_failures", 0)),
        "optical_fallbacks": int(result.get("optical_fallbacks", 0)),
        "wall_s": wall_s,
        "cost_identity_error_s": float(env.virtual_time - reconstructed),
    }


def bootstrap_ci(delta: np.ndarray, seed: int = 20260912) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(delta), size=(10000, len(delta)))
    means = delta[samples].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(rows: list[dict]) -> dict:
    groups = []
    for error_mode in sorted({r["error_mode"] for r in rows}):
        for mode in (3, 4):
            subset = [r for r in rows if r["error_mode"] == error_mode and r["mode"] == mode]
            if not subset:
                continue
            by_variant = {
                name: sorted((r for r in subset if r["variant"] == name), key=lambda r: r["seed"])
                for name in ("G25OR", "ISR")
            }
            base = by_variant["G25OR"]
            candidate = by_variant["ISR"]
            assert [r["seed"] for r in base] == [r["seed"] for r in candidate]
            b = np.asarray([r["seconds_per_source"] for r in base])
            c = np.asarray([r["seconds_per_source"] for r in candidate])
            groups.append({
                "error_mode": error_mode, "mode": mode, "pairs": len(base),
                "all_clear": bool(all(r["success"] and r["cleared"] == r["sources"] for r in subset)),
                "baseline_mean": float(b.mean()), "isr_mean": float(c.mean()),
                "relative_change_percent": float((c.mean() / b.mean() - 1.0) * 100.0),
                "paired_mean_delta_ci95": bootstrap_ci(c - b),
                "baseline_median": float(np.median(b)), "isr_median": float(np.median(c)),
                "baseline_p90": float(np.quantile(b, 0.90)), "isr_p90": float(np.quantile(c, 0.90)),
                "baseline_p95": float(np.quantile(b, 0.95)), "isr_p95": float(np.quantile(c, 0.95)),
                "baseline_max": float(b.max()), "isr_max": float(c.max()),
                "wins": int(np.sum(c < b - 1e-7)), "losses": int(np.sum(c > b + 1e-7)),
            })
    max_cost_error = max(abs(r["cost_identity_error_s"]) for r in rows)
    return {"groups": groups, "runs": len(rows), "max_cost_identity_error_s": max_cost_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "isr_validation" / "local"))
    parser.add_argument("--holdout", type=int, default=200)
    parser.add_argument("--stress", type=int, default=30)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for mode in (3, 4):
        for offset in range(args.holdout):
            jobs.append((mode, 260000 + offset, "edge" if offset % 3 == 0 else "uniform", "spatial"))
        for offset in range(args.stress):
            jobs.append((mode, 270000 + offset, "edge" if offset % 3 == 0 else "uniform", "endpoints"))
            jobs.append((mode, 270100 + offset, "edge" if offset % 3 == 0 else "uniform", "constant"))

    rows = []
    for index, (mode, seed, kind, error_mode) in enumerate(jobs, 1):
        for variant in ("G25OR", "ISR"):
            row = run_one(mode, seed, kind, variant, error_mode)
            rows.append(row)
            if not row["success"] or row["cleared"] != row["sources"]:
                raise RuntimeError(f"not all sources cleared: {row}")
            if abs(row["cost_identity_error_s"]) > 1e-6:
                raise RuntimeError(f"cost identity failed: {row}")
        if index % 25 == 0:
            print(f"[local] scenarios={index}/{len(jobs)} runs={len(rows)}", flush=True)

    with (out / "runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    metadata = {
        "git_head": git_head(), "python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "bearing_decimals": 2,
        "code_sha256": {
            "brl/independent_candidate.py": sha256(ROOT / "brl" / "independent_candidate.py"),
            "brl/bilateral.py": sha256(ROOT / "brl" / "bilateral.py"),
            "brl/local_env.py": sha256(ROOT / "brl" / "local_env.py"),
            "scripts/run_isr_local_validation.py": sha256(Path(__file__).resolve()),
        },
        "holdout_seed_range": [260000, 260000 + args.holdout - 1],
        "endpoint_seed_range": [270000, 270000 + args.stress - 1],
        "constant_seed_range": [270100, 270100 + args.stress - 1],
        "bootstrap_samples": 10000, "bootstrap_seed": 20260912,
    }
    (out / "summary.json").write_text(
        json.dumps({"metadata": metadata, **summary}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
