"""Incremental paired evaluation for ISR-v1, JSP-G and JSP-L."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn

from brl.independent_candidate import IndependentCandidate
from brl.jsp import JSPCandidate, JSPConfig
from brl.jsp.model import JSPRanker
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources


FIELDS = ["mode", "seed", "kind", "error_mode", "arm", "sources", "cleared",
          "success", "virtual_time_s", "seconds_per_source", "distance_m",
          "measure_calls", "switches", "clear_calls", "failed_clear",
          "planning_wall_s", "planning_disabled_reason", "optical_fallbacks",
          "wall_s", "cost_identity_error_s", "status",
          "error_type", "error"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_manifest(out: Path, args, modes: list[int], arms: list[str]) -> None:
    files = [
        Path(__file__).resolve(), ROOT / "scripts" / "run_g25o_compare.py",
        ROOT / "brl" / "jsp" / "policy.py", ROOT / "brl" / "jsp" / "snapshot.py",
        ROOT / "brl" / "jsp" / "model.py", ROOT / "brl" / "bilateral_state.py",
        ROOT / "brl" / "bilateral.py", ROOT / "brl" / "independent_candidate.py",
        ROOT / "brl" / "local_env.py", ROOT / "brl" / "protocol.py",
    ]
    models = {}
    for mode, value in ((3, args.model_q3), (4, args.model_q4)):
        if value:
            model = Path(value).resolve(); metadata = model.with_suffix(model.suffix + ".json")
            models[str(mode)] = {
                "path": str(model), "sha256": sha256(model),
                "metadata_sha256": sha256(metadata),
            }
    manifest = {
        "schema_version": 1, "stage": args.stage, "modes": modes, "arms": arms,
        "models": models, "python": platform.python_version(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "bootstrap_seed": 20260912,
        "git_head_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed_audit": ("160000-160049 is development" if args.stage == "dev" else
                       "280000-280199, 281000-281029, 281100-281129 checked unused before JSP acceptance"),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
    }
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("resume manifest does not match frozen JSP experiment")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


class EndpointErrorField:
    def value(self, channel, x, y):
        return 1.0 if np.sin(0.01 * x + 0.013 * y + channel) > 0 else -1.0


class ConstantErrorField:
    def value(self, channel, x, y):
        return 1.0


def run_one(mode: int, seed: int, kind: str, error_mode: str, arm: str,
            model_path: str = "") -> tuple[dict, list[dict]]:
    sources = generate_sources(seed, mode, kind)
    env = RadioEnv(mode=mode, n_sources=len(sources), seed=seed,
                   step_limit=20000, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    if error_mode == "endpoints":
        env.error_field = EndpointErrorField()
    elif error_mode == "constant":
        env.error_field = ConstantErrorField()
    elif error_mode != "spatial":
        raise ValueError(error_mode)
    started = time.perf_counter()
    if arm == "ISR":
        result = IndependentCandidate(mode).run(env)
    elif arm == "JSPG":
        result = JSPCandidate(mode, planner="analytic").run(env)
    elif arm == "JSPBASE":
        result = JSPCandidate(mode, config=JSPConfig(detour_limit_m=-1.0)).run(env)
    elif arm == "JSPL":
        result = JSPCandidate(mode, planner="learned", model_path=model_path).run(env)
    else:
        raise ValueError(arm)
    wall = time.perf_counter() - started
    n = len(sources)
    reconstructed = (env.move_distance / 5.0 + 5.0 * env.n_measure + env.n_switch
                     + 3.0 * env.n_clear_fail + 5.0 * env.cleared_count())
    row = {
        "mode": mode, "seed": seed, "kind": kind, "error_mode": error_mode,
        "arm": arm, "sources": n, "cleared": env.cleared_count(),
        "success": bool(result["success"] and env.completion_certificate()),
        "virtual_time_s": env.virtual_time, "seconds_per_source": env.virtual_time / n,
        "distance_m": env.move_distance, "measure_calls": env.n_measure,
        "switches": env.n_switch, "clear_calls": env.n_clear,
        "failed_clear": env.n_clear_fail,
        "planning_wall_s": result.get("planning_wall_s", 0.0), "wall_s": wall,
        "planning_disabled_reason": result.get("planning_disabled_reason", ""),
        "optical_fallbacks": result.get("optical_fallbacks", 0),
        "cost_identity_error_s": env.virtual_time - reconstructed,
        "status": "complete", "error_type": "", "error": "",
    }
    decisions = [{"mode": mode, "seed": seed, "arm": arm, **d}
                 for d in result.get("decision_log", [])]
    return row, decisions


def jobs(stage: str, modes: list[int], arms: list[str]):
    if stage == "dev":
        groups = ((160000, 50, "spatial"),)
    elif stage == "acceptance":
        groups = ((280000, 200, "spatial"), (281000, 30, "endpoints"),
                  (281100, 30, "constant"))
    else:
        raise ValueError(stage)
    for mode in modes:
        for start, count, error in groups:
            for offset in range(count):
                seed = start + offset
                kind = "edge" if offset % 3 == 0 else "uniform"
                for arm in arms:
                    yield mode, seed, kind, error, arm


def summarize(rows: list[dict], stage: str) -> dict:
    out = []
    for mode in sorted({int(r["mode"]) for r in rows}):
        for error in sorted({r["error_mode"] for r in rows if int(r["mode"]) == mode}):
            group = [r for r in rows if int(r["mode"]) == mode and r["error_mode"] == error]
            base = {int(r["seed"]): r for r in group if r["arm"] == "ISR"}
            for arm in sorted({r["arm"] for r in group if r["arm"] != "ISR"}):
                cand = {int(r["seed"]): r for r in group if r["arm"] == arm}
                seeds = sorted(set(base) & set(cand))
                b = np.asarray([float(base[s]["seconds_per_source"]) for s in seeds])
                c = np.asarray([float(cand[s]["seconds_per_source"]) for s in seeds])
                delta = c - b
                rng = np.random.default_rng(20260912)
                idx = rng.integers(0, len(delta), size=(10000, len(delta)))
                bootstrap = delta[idx].mean(axis=1)
                ci = np.quantile(bootstrap, [0.025, 0.975])
                ci975 = np.quantile(bootstrap, [0.0125, 0.9875])
                out.append({
                    "mode": mode, "error_mode": error, "candidate": arm, "pairs": len(seeds),
                    "all_clear": all(str(r["success"]).lower() == "true" and int(r["sources"]) == int(r["cleared"]) for r in group),
                    "baseline_mean": float(b.mean()), "candidate_mean": float(c.mean()),
                    "relative_change_percent": float((c.mean() / b.mean() - 1) * 100),
                    "paired_delta_ci95": [float(x) for x in ci],
                    "paired_delta_ci97_5": [float(x) for x in ci975],
                    "baseline_p95": float(np.quantile(b, .95)), "candidate_p95": float(np.quantile(c, .95)),
                    "baseline_max": float(b.max()), "candidate_max": float(c.max()),
                    "baseline_wall_p95_s": float(np.quantile(
                        [float(base[s]["wall_s"]) for s in seeds], .95)),
                    "candidate_wall_p95_s": float(np.quantile(
                        [float(cand[s]["wall_s"]) for s in seeds], .95)),
                    "mean_distance_delta_m": float(np.mean([float(cand[s]["distance_m"]) - float(base[s]["distance_m"]) for s in seeds])),
                    "mean_measure_delta": float(np.mean([int(cand[s]["measure_calls"]) - int(base[s]["measure_calls"]) for s in seeds])),
                    "mean_switch_delta": float(np.mean([int(cand[s]["switches"]) - int(base[s]["switches"]) for s in seeds])),
                })
    summary = {"runs": len(rows),
               "max_cost_identity_error_s": max(abs(float(r["cost_identity_error_s"])) for r in rows),
               "groups": out}
    if stage == "dev":
        selections = []
        for mode in sorted({item["mode"] for item in out}):
            items = {item["candidate"]: item for item in out
                     if item["mode"] == mode and item["error_mode"] == "spatial"}
            eligible = [name for name, item in items.items()
                        if item["all_clear"] and item["relative_change_percent"] <= -3.0
                        and item["candidate_p95"] <= 1.03 * item["baseline_p95"]
                        and item["candidate_max"] <= 1.03 * item["baseline_max"]]
            preferred = None
            if "JSPG" in eligible:
                preferred = "JSPG"
            if "JSPL" in eligible:
                if ("JSPG" not in items
                        or items["JSPL"]["candidate_mean"] <= .99 * items["JSPG"]["candidate_mean"]):
                    preferred = "JSPL"
                elif preferred is None:
                    preferred = "JSPG" if "JSPG" in eligible else None
            selections.append({"mode": mode, "eligible": eligible, "preferred": preferred,
                               "learned_extra_one_percent": bool(
                                   "JSPL" in items and "JSPG" in items and
                                   items["JSPL"]["candidate_mean"] <= .99 * items["JSPG"]["candidate_mean"] )})
        summary["development_selection"] = selections
    if stage == "acceptance":
        gates = []
        for mode in sorted({item["mode"] for item in out}):
            for candidate in sorted({item["candidate"] for item in out if item["mode"] == mode}):
                items = {item["error_mode"]: item for item in out
                         if item["mode"] == mode and item["candidate"] == candidate}
                main = items.get("spatial")
                if main is None:
                    continue
                checks = {
                    "all_clear": all(item["all_clear"] for item in items.values()),
                    "cost_identity": summary["max_cost_identity_error_s"] <= 1e-6,
                    "main_mean_improves_3pct": main["relative_change_percent"] <= -3.0,
                    "main_ci97_5_upper_below_zero": main["paired_delta_ci97_5"][1] < 0.0,
                    "p95_within_103pct": main["candidate_p95"] <= 1.03 * main["baseline_p95"],
                    "max_within_103pct": main["candidate_max"] <= 1.03 * main["baseline_max"],
                    "stress_means_within_3pct": all(
                        item["relative_change_percent"] <= 3.0 for name, item in items.items()
                        if name != "spatial"),
                    "runtime_p95_within_limit": main["candidate_wall_p95_s"] <= max(
                        2.0 * main["baseline_wall_p95_s"], 10.0),
                }
                gates.append({"mode": mode, "candidate": candidate, "checks": checks,
                              "accepted": all(checks.values())})
        summary["acceptance_gates"] = gates
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "acceptance"], required=True)
    parser.add_argument("--modes", default="3,4")
    parser.add_argument("--arms", default="ISR,JSPG")
    parser.add_argument("--model-q3", default="")
    parser.add_argument("--model-q4", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    modes = [int(x) for x in args.modes.split(",")]
    arms = [x.strip().upper() for x in args.arms.split(",")]
    for mode in modes:
        if "JSPL" in arms:
            model = args.model_q3 if mode == 3 else args.model_q4
            if not model:
                raise SystemExit(f"JSPL mode {mode} requires its --model-q{mode} path")
            JSPRanker(model, mode)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    freeze_manifest(out, args, modes, arms)
    runs_path = out / "runs.csv"; decisions_path = out / "decisions.jsonl"
    existing = []
    done = set()
    if runs_path.exists():
        with runs_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
        done = {(int(r["mode"]), int(r["seed"]), r["error_mode"], r["arm"]) for r in existing}
    write_header = not runs_path.exists()
    with runs_path.open("a", encoding="utf-8-sig" if write_header else "utf-8",
                        newline="") as run_handle, decisions_path.open("a", encoding="utf-8") as decision_handle:
        writer = csv.DictWriter(run_handle, fieldnames=FIELDS)
        if write_header: writer.writeheader()
        all_jobs = list(jobs(args.stage, modes, arms))
        for index, (mode, seed, kind, error, arm) in enumerate(all_jobs, 1):
            if (mode, seed, error, arm) in done:
                continue
            model = args.model_q3 if mode == 3 else args.model_q4
            try:
                row, logs = run_one(mode, seed, kind, error, arm, model)
            except Exception as exc:
                failure = {
                    "mode": mode, "seed": seed, "kind": kind, "error_mode": error,
                    "arm": arm, "sources": "", "cleared": "", "success": False,
                    "virtual_time_s": "", "seconds_per_source": "", "distance_m": "",
                    "measure_calls": "", "switches": "", "clear_calls": "",
                    "failed_clear": "", "planning_wall_s": "",
                    "planning_disabled_reason": "", "optical_fallbacks": "", "wall_s": "",
                    "cost_identity_error_s": "", "status": "failed",
                    "error_type": type(exc).__name__, "error": str(exc),
                }
                writer.writerow(failure); run_handle.flush()
                with (out / "failures.jsonl").open("a", encoding="utf-8") as failures:
                    failures.write(json.dumps(failure, ensure_ascii=False) + "\n")
                raise
            writer.writerow(row); run_handle.flush()
            for event in logs:
                decision_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            decision_handle.flush()
            if not row["success"] or row["sources"] != row["cleared"] or abs(row["cost_identity_error_s"]) > 1e-6:
                raise RuntimeError(f"JSP experiment gate failed: {row}")
            if index % 20 == 0 or index == len(all_jobs):
                print(f"runs={index}/{len(all_jobs)}", flush=True)
    with runs_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(r.get("status") != "complete" for r in rows):
        raise RuntimeError("experiment contains a failed persisted run")
    summary = summarize(rows, args.stage)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
