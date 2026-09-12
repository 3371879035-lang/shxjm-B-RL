"""Reproducible ISR-v2 development and frozen paired-validation harness."""
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

from brl.independent_candidate import IndependentCandidate, IndependentCandidateNoProbe
from brl.isr_v2 import ISRV2Candidate, ISRV2Config
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources


class EndpointErrorField:
    def value(self, channel: int, x: float, y: float) -> float:
        return 1.0 if np.sin(0.01 * x + 0.013 * y + channel) > 0 else -1.0


class ConstantErrorField:
    def value(self, channel: int, x: float, y: float) -> float:
        return 1.0


ARM_CONFIGS = {
    "SAFE": None,
    "NOPROBE": "noprobe",
    "SOFT": ISRV2Config(soft_probe=True, multistart_route=False, scan_reprice=False, q4_signal_model=False),
    "MULTI": ISRV2Config(soft_probe=False, multistart_route=True, scan_reprice=False, q4_signal_model=False),
    "REPRICE": ISRV2Config(soft_probe=False, multistart_route=False, scan_reprice=True, q4_signal_model=True),
    "COMBO": ISRV2Config(soft_probe=True, multistart_route=True, scan_reprice=True, q4_signal_model=True),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy(mode: int, arm: str):
    cfg = ARM_CONFIGS[arm]
    if arm == "SAFE":
        return IndependentCandidate(mode)
    if arm == "NOPROBE":
        return IndependentCandidateNoProbe(mode)
    return ISRV2Candidate(mode, cfg)


def run_one(mode: int, seed: int, kind: str, error_mode: str, arm: str) -> tuple[dict, list[dict]]:
    sources = generate_sources(seed, mode, kind)
    env = RadioEnv(mode=mode, n_sources=len(sources), seed=seed, step_limit=20000, bearing_decimals=2)
    env.reset(seed=seed, n_sources=len(sources), sources=sources)
    if error_mode == "endpoints":
        env.error_field = EndpointErrorField()
    elif error_mode == "constant":
        env.error_field = ConstantErrorField()
    elif error_mode != "spatial":
        raise ValueError(error_mode)
    started = time.perf_counter()
    result = _policy(mode, arm).run(env)
    wall_s = time.perf_counter() - started
    n = len(sources)
    reconstructed = env.move_distance / 5.0 + 5.0 * env.n_measure + env.n_switch + 3.0 * env.n_clear_fail + 5.0 * env.cleared_count()
    row = {
        "mode": mode, "seed": seed, "kind": kind, "error_mode": error_mode, "arm": arm,
        "sources": n, "cleared": env.cleared_count(),
        "success": bool(result.get("success") and env.completion_certificate()),
        "virtual_time_s": float(env.virtual_time), "seconds_per_source": float(env.virtual_time / n),
        "distance_m": float(env.move_distance), "measure_calls": int(env.n_measure),
        "switches": int(env.n_switch), "clear_calls": int(env.n_clear),
        "failed_clear": int(env.n_clear_fail), "solver_failures": int(result.get("solver_failures", 0)),
        "optical_fallbacks": int(result.get("optical_fallbacks", 0)),
        "soft_actions": int(result.get("soft_actions", 0)),
        "planning_wall_s": float(result.get("planning_wall_s", 0.0)),
        "soft_disabled_reason": str(result.get("soft_disabled_reason", "")),
        "wall_s": float(wall_s), "cost_identity_error_s": float(env.virtual_time - reconstructed),
    }
    decisions = [{"mode": mode, "seed": seed, "arm": arm, **d} for d in result.get("decision_log", [])]
    return row, decisions


def _bootstrap(delta: np.ndarray, confidence: float, seed: int = 20260912) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(10000, len(delta)))
    means = delta[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [float(x) for x in np.quantile(means, [alpha, 1.0 - alpha])]


def summarize(rows: list[dict], confidence: float) -> dict:
    groups = []
    for mode in sorted({r["mode"] for r in rows}):
        for error_mode in sorted({r["error_mode"] for r in rows if r["mode"] == mode}):
            subset = [r for r in rows if r["mode"] == mode and r["error_mode"] == error_mode]
            base = sorted((r for r in subset if r["arm"] == "SAFE"), key=lambda x: x["seed"])
            for arm in sorted({r["arm"] for r in subset if r["arm"] != "SAFE"}):
                cand = sorted((r for r in subset if r["arm"] == arm), key=lambda x: x["seed"])
                if [r["seed"] for r in base] != [r["seed"] for r in cand]:
                    raise RuntimeError(f"unpaired rows mode={mode} error={error_mode} arm={arm}")
                b = np.asarray([r["seconds_per_source"] for r in base])
                c = np.asarray([r["seconds_per_source"] for r in cand])
                bw = np.asarray([r["wall_s"] for r in base])
                cw = np.asarray([r["wall_s"] for r in cand])
                groups.append({
                    "mode": mode, "error_mode": error_mode, "candidate": arm, "pairs": len(base),
                    "all_clear": bool(all(r["success"] and r["sources"] == r["cleared"] for r in base + cand)),
                    "baseline_mean": float(b.mean()), "candidate_mean": float(c.mean()),
                    "relative_change_percent": float((c.mean() / b.mean() - 1.0) * 100.0),
                    f"paired_delta_ci{confidence * 100:g}": _bootstrap(c - b, confidence),
                    "baseline_p90": float(np.quantile(b, 0.90)), "candidate_p90": float(np.quantile(c, 0.90)),
                    "baseline_p95": float(np.quantile(b, 0.95)), "candidate_p95": float(np.quantile(c, 0.95)),
                    "baseline_max": float(b.max()), "candidate_max": float(c.max()),
                    "baseline_wall_p95": float(np.quantile(bw, 0.95)), "candidate_wall_p95": float(np.quantile(cw, 0.95)),
                    "wins": int(np.sum(c < b - 1e-9)), "ties": int(np.sum(np.abs(c - b) <= 1e-9)),
                    "losses": int(np.sum(c > b + 1e-9)),
                    "mean_distance_delta_m": float(np.mean([r["distance_m"] for r in cand]) - np.mean([r["distance_m"] for r in base])),
                    "mean_measure_delta": float(np.mean([r["measure_calls"] for r in cand]) - np.mean([r["measure_calls"] for r in base])),
                    "mean_switch_delta": float(np.mean([r["switches"] for r in cand]) - np.mean([r["switches"] for r in base])),
                    "failed_clear_baseline": int(sum(r["failed_clear"] for r in base)),
                    "failed_clear_candidate": int(sum(r["failed_clear"] for r in cand)),
                    "soft_budget_fallbacks": int(sum(bool(r["soft_disabled_reason"]) for r in cand)),
                })
    return {"runs": len(rows), "max_cost_identity_error_s": max(abs(r["cost_identity_error_s"]) for r in rows), "groups": groups}


def _jobs(stage: str, modes: list[int], arms: list[str], q3: str, q4: str):
    if stage == "dev":
        for mode in modes:
            for offset in range(50):
                seed = 160000 + offset
                kind = "edge" if offset % 3 == 0 else "uniform"
                for arm in arms:
                    yield mode, seed, kind, "spatial", arm
    elif stage == "acceptance":
        for mode in modes:
            candidate = q3 if mode == 3 else q4
            for start, count, error in ((280000, 200, "spatial"), (281000, 30, "endpoints"), (281100, 30, "constant")):
                for offset in range(count):
                    seed = start + offset
                    kind = "edge" if offset % 3 == 0 else "uniform"
                    for arm in ("SAFE", candidate):
                        yield mode, seed, kind, error, arm
    else:
        raise ValueError(stage)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "acceptance"], required=True)
    ap.add_argument("--modes", default="3,4")
    ap.add_argument("--arms", default="SAFE,NOPROBE,SOFT,MULTI,REPRICE,COMBO")
    ap.add_argument("--candidate-q3", default="COMBO", choices=list(ARM_CONFIGS))
    ap.add_argument("--candidate-q4", default="COMBO", choices=list(ARM_CONFIGS))
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    modes = [int(x) for x in args.modes.split(",") if x]
    arms = [x.strip().upper() for x in args.arms.split(",") if x.strip()]
    if "SAFE" not in arms and args.stage == "dev":
        raise SystemExit("development arms must include SAFE")
    for arm in arms:
        if arm not in ARM_CONFIGS:
            raise SystemExit(f"unknown arm {arm}")
    out = Path(args.out_dir) if args.out_dir else ROOT / "results" / "isrv2" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    jobs = list(_jobs(args.stage, modes, arms, args.candidate_q3, args.candidate_q4))
    rows: list[dict] = []
    decisions: list[dict] = []
    for index, job in enumerate(jobs, 1):
        row, logs = run_one(*job)
        rows.append(row); decisions.extend(logs)
        if not row["success"] or row["cleared"] != row["sources"]:
            raise RuntimeError(f"completion check failed: {row}")
        if abs(row["cost_identity_error_s"]) > 1e-6:
            raise RuntimeError(f"cost identity failed: {row}")
        if index % 50 == 0 or index == len(jobs):
            print(f"[{args.stage}] runs={index}/{len(jobs)}", flush=True)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    confidence = 0.95 if args.stage == "dev" else 0.975
    summary = summarize(rows, confidence)
    metadata = {
        "stage": args.stage, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "bearing_decimals": 2, "bootstrap_samples": 10000, "bootstrap_seed": 20260912,
        "modes": modes, "arms": arms if args.stage == "dev" else {"3": args.candidate_q3, "4": args.candidate_q4},
        "seed_ranges": {"dev": [160000, 160049]} if args.stage == "dev" else {
            "spatial": [280000, 280199], "endpoints": [281000, 281029], "constant": [281100, 281129]},
        "code_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in [
            ROOT / "brl" / "independent_candidate.py", ROOT / "brl" / "isr_v2.py",
            ROOT / "brl" / "bilateral.py", Path(__file__).resolve()]},
    }
    (out / "summary.json").write_text(json.dumps({"metadata": metadata, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
