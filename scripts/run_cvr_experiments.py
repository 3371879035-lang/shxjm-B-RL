"""Resumable paired local evaluation for ISR, fixed segments, and CVR."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
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
import scipy

from brl.coverage import s25_points, s3_points
from brl.cvr import CVRCandidate, CVRConfig
from brl.cvr.journal import EventJournal, JournaledActionView
from brl.independent_candidate import CheckedView, IndependentCandidate
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources


MECHANISM_FIELDS = [
    "replacement_candidates",
    "certified_replacements",
    "executed_replacements",
    "restored_legacy_stations",
    "physical_stations_removed",
    "extra_measurements",
    "replans",
    "segment_interruptions",
    "certificate_wall_s",
    "fallback_reason",
    "fixed_obligations_completed", "variable_obligations_completed",
    "fixed_stations_visited", "variable_points_visited", "repeat_station_visits",
    "declared_exits_visited", "mean_planned_depth", "locator_action_events",
]

FIELDS = [
    "mode", "seed", "kind", "error_mode", "arm", "sources", "cleared", "success",
    "virtual_time_s", "seconds_per_source", "distance_m", "measure_calls", "switches",
    "clear_calls", "failed_clear", "planning_wall_s", "wall_s", "cost_identity_error_s",
    *MECHANISM_FIELDS, "code_hash", "config_hash", "status", "error_type", "error",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def code_hash() -> str:
    files = sorted((ROOT / "brl" / "cvr").glob("*.py")) + [
        ROOT / "brl" / "bilateral.py", ROOT / "brl" / "bilateral_state.py",
        ROOT / "brl" / "coverage.py", ROOT / "brl" / "geometry.py",
        ROOT / "brl" / "independent_candidate.py", ROOT / "brl" / "jsp" / "policy.py",
        ROOT / "brl" / "local_env.py", ROOT / "brl" / "protocol.py",
        ROOT / "scripts" / "run_g25o_compare.py", Path(__file__).resolve(),
    ]
    return _stable_hash({str(path.relative_to(ROOT)): _sha256(path) for path in files})


def arm_config(mode: int, arm: str) -> dict:
    if arm == "ISR":
        return {"candidate": "IndependentCandidate", "probe": True}
    if arm == "COMPAT":
        return asdict(CVRConfig(planner_enabled=False))
    if arm == "SEGFIXED":
        return asdict(CVRConfig(variable_waypoints=False))
    if arm == "CVR":
        return asdict(CVRConfig())
    raise ValueError(arm)


def config_hash(mode: int, arm: str) -> str:
    return _stable_hash({"mode": mode, "arm": arm, "bearing_decimals": 2,
                         "config": arm_config(mode, arm)})


class EndpointErrorField:
    def value(self, channel, x, y):
        return 1.0 if np.sin(0.01 * x + 0.013 * y + channel) > 0 else -1.0


class ConstantErrorField:
    def value(self, channel, x, y):
        return 1.0


def jobs(stage: str, modes: list[int], arms: list[str]):
    groups = {
        "dev": ((160000, 50, "spatial"),),
        "acceptance": (
            (280000, 200, "spatial"),
            (281000, 30, "endpoints"),
            (281100, 30, "constant"),
        ),
    }[stage]
    for mode in modes:
        for start, count, error_mode in groups:
            for offset in range(count):
                seed = start + offset
                kind = "edge" if offset % 3 == 0 else "uniform"
                for arm in arms:
                    yield mode, seed, kind, error_mode, arm


def make_candidate(mode: int, arm: str):
    if arm == "ISR":
        return IndependentCandidate(mode)
    if arm == "COMPAT":
        return CVRCandidate(mode, CVRConfig(planner_enabled=False))
    if arm == "SEGFIXED":
        return CVRCandidate(mode, CVRConfig(variable_waypoints=False))
    if arm == "CVR":
        return CVRCandidate(mode)
    raise ValueError(arm)


def _initial_event(mode: int) -> dict:
    points = s3_points() if mode == 3 else s25_points()
    return {
        "event": "plan_initial", "mode": mode, "plan_version": 1,
        "nodes": [{"node_id": f"fixed-{index}", "position": list(map(float, point)),
                   "channels": list(range(1, 21)), "legacy_station": index}
                  for index, point in enumerate(points)],
    }


def _mechanism_from_events(events: list[dict]) -> dict:
    measurements = [row for row in events
                    if row.get("event") == "action_confirmed"
                    and row.get("action_kind") == "measure"]
    fixed = [row for row in measurements if row.get("coverage_idx") is not None]
    variable = [row for row in measurements
                if row.get("coverage_idx") is None and not row.get("is_refine")]
    fixed_sequence = [int(row["coverage_idx"]) for row in fixed]
    station_arrivals = [value for index, value in enumerate(fixed_sequence)
                        if index == 0 or fixed_sequence[index - 1] != value]
    repeat_visits = len(station_arrivals) - len(set(station_arrivals))
    variable_points = {(round(float(row["position"][0]), 8),
                        round(float(row["position"][1]), 8)) for row in variable}
    segments = [row for row in events if row.get("event") == "segment"]
    actions = [row for row in events if row.get("event") == "action_confirmed"]
    exits_visited = 0
    for index, segment in enumerate(segments):
        start_seq = int(segment.get("sequence", 0))
        end_seq = int(segments[index + 1].get("sequence", 10**18)) if index + 1 < len(segments) else 10**18
        exit_point = np.asarray(segment.get("declared_exit_position", [np.inf, np.inf]), dtype=float)
        if any(start_seq < int(action.get("sequence", 0)) < end_seq
               and np.linalg.norm(np.asarray(action.get("position"), dtype=float) - exit_point) <= 1e-8
               for action in actions):
            exits_visited += 1
    return {
        "fixed_obligations_completed": len(fixed),
        "variable_obligations_completed": len(variable),
        "fixed_stations_visited": len(set(fixed_sequence)),
        "variable_points_visited": len(variable_points),
        "repeat_station_visits": repeat_visits,
        "declared_exits_visited": exits_visited,
        "mean_planned_depth": float(np.mean([row.get("planned_depth", 0) for row in segments])) if segments else 0.0,
        "locator_action_events": sum(row.get("event") == "locator_action" for row in events),
    }


def run_one(mode: int, seed: int, kind: str, error_mode: str, arm: str,
            journal_path: Path | None = None) -> tuple[dict, list[dict]]:
    sources = generate_sources(seed, mode, kind)
    env = RadioEnv(
        mode=mode,
        n_sources=len(sources),
        seed=seed,
        step_limit=20000,
        bearing_decimals=2,
    )
    env.reset(seed=seed, sources=sources)
    if error_mode == "endpoints":
        env.error_field = EndpointErrorField()
    elif error_mode == "constant":
        env.error_field = ConstantErrorField()
    elif error_mode != "spatial":
        raise ValueError(error_mode)

    started = time.perf_counter()
    if arm == "ISR":
        journal = EventJournal(journal_path)
        journal.emit(_initial_event(mode))
        wrapped = JournaledActionView(CheckedView(env), journal)
        result = make_candidate(mode, arm).run(wrapped)
        success = bool(result["success"] and env.completion_certificate())
        journal.emit({
            "event": "terminal", "status": "complete" if success else "incomplete",
            "success": success, "completion_certificate": success,
            "cleared": env.cleared_count(), "virtual_time_s": env.virtual_time,
            "distance_m": env.move_distance, "measure_calls": env.n_measure,
            "switches": env.n_switch, "clear_calls": env.n_clear,
            "failed_clear": env.n_clear_fail, "plan_version": 1,
        })
        result["decision_log"] = list(journal.events)
    else:
        result = make_candidate(mode, arm).run(env, journal_path=journal_path)
    wall_s = time.perf_counter() - started
    source_count = len(sources)
    reconstructed = (
        env.move_distance / 5.0
        + 5.0 * env.n_measure
        + env.n_switch
        + 3.0 * env.n_clear_fail
        + 5.0 * env.cleared_count()
    )
    revision = code_hash()
    config = config_hash(mode, arm)
    mechanism = _mechanism_from_events(result.get("decision_log", []))
    row = {
        "mode": mode,
        "seed": seed,
        "kind": kind,
        "error_mode": error_mode,
        "arm": arm,
        "sources": source_count,
        "cleared": env.cleared_count(),
        "success": bool(result["success"] and env.cleared_count() == source_count),
        "virtual_time_s": env.virtual_time,
        "seconds_per_source": env.virtual_time / source_count,
        "distance_m": env.move_distance,
        "measure_calls": env.n_measure,
        "switches": env.n_switch,
        "clear_calls": env.n_clear,
        "failed_clear": env.n_clear_fail,
        "planning_wall_s": result.get("planning_wall_s", 0.0),
        "wall_s": wall_s,
        "cost_identity_error_s": env.virtual_time - reconstructed,
        "replacement_candidates": result.get("replacement_candidates", 0),
        "certified_replacements": result.get("certified_replacements", 0),
        "executed_replacements": result.get("executed_replacements", 0),
        "restored_legacy_stations": result.get("restored_legacy_stations", 0),
        "physical_stations_removed": result.get("physical_stations_removed", 0),
        "extra_measurements": result.get("extra_measurements", 0),
        "replans": result.get("replans", 0),
        "segment_interruptions": result.get("segment_interruptions", 0),
        "certificate_wall_s": result.get("planning_wall_s", 0.0),
        "fallback_reason": result.get("fallback_reason", ""),
        **mechanism,
        "code_hash": revision,
        "config_hash": config,
        "status": "complete",
        "error_type": "",
        "error": "",
    }
    decisions = [
        {"mode": mode, "seed": seed, "kind": kind, "error_mode": error_mode, "arm": arm, **event}
        for event in result.get("decision_log", [])
    ]
    return row, decisions


def _complete_rows(rows: list[dict]) -> list[dict]:
    complete: dict[tuple, dict] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = (int(row["mode"]), int(row["seed"]), row["error_mode"], row["arm"])
        complete[key] = row
    return list(complete.values())


def _truth(value) -> bool:
    return value is True or str(value).lower() == "true"


def _bootstrap(delta: np.ndarray) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(20260912)
    indexes = rng.integers(0, len(delta), size=(10000, len(delta)))
    means = delta[indexes].mean(axis=1)
    return (
        [float(value) for value in np.quantile(means, [0.025, 0.975])],
        [float(value) for value in np.quantile(means, [0.0125, 0.9875])],
    )


def summarize(rows: list[dict], stage: str) -> dict:
    rows = _complete_rows(rows)
    groups = []
    for mode in sorted({int(row["mode"]) for row in rows}):
        errors = sorted({row["error_mode"] for row in rows if int(row["mode"]) == mode})
        for error_mode in errors:
            subset = [row for row in rows if int(row["mode"]) == mode and row["error_mode"] == error_mode]
            baseline = {int(row["seed"]): row for row in subset if row["arm"] == "ISR"}
            for arm in sorted({row["arm"] for row in subset if row["arm"] != "ISR"}):
                candidate = {int(row["seed"]): row for row in subset if row["arm"] == arm}
                seeds = sorted(set(baseline) & set(candidate))
                if not seeds:
                    continue
                b = np.asarray([float(baseline[seed]["seconds_per_source"]) for seed in seeds])
                c = np.asarray([float(candidate[seed]["seconds_per_source"]) for seed in seeds])
                delta = c - b
                ci95, ci975 = _bootstrap(delta)
                all_rows = [baseline[seed] for seed in seeds] + [candidate[seed] for seed in seeds]
                mechanism = {
                    field: float(np.mean([float(candidate[seed].get(field, 0) or 0) for seed in seeds]))
                    for field in MECHANISM_FIELDS
                    if field != "fallback_reason"
                }
                distance_delta = float(np.mean([
                    float(candidate[seed]["distance_m"]) - float(baseline[seed]["distance_m"])
                    for seed in seeds
                ]))
                mean_delta = float(delta.mean())
                group = {
                    "mode": mode,
                    "error_mode": error_mode,
                    "candidate": arm,
                    "pairs": len(seeds),
                    "all_clear": all(
                        _truth(row["success"]) and int(row["sources"]) == int(row["cleared"])
                        for row in all_rows
                    ),
                    "cost_identity_ok": all(abs(float(row["cost_identity_error_s"])) <= 1e-6 for row in all_rows),
                    "baseline_mean": float(b.mean()),
                    "candidate_mean": float(c.mean()),
                    "mean_delta_s_per_source": mean_delta,
                    "relative_change_percent": float((c.mean() / b.mean() - 1.0) * 100.0),
                    "paired_delta_ci95": ci95,
                    "paired_delta_ci97_5": ci975,
                    "baseline_p90": float(np.quantile(b, 0.90)),
                    "candidate_p90": float(np.quantile(c, 0.90)),
                    "baseline_p95": float(np.quantile(b, 0.95)),
                    "candidate_p95": float(np.quantile(c, 0.95)),
                    "baseline_max": float(b.max()),
                    "candidate_max": float(c.max()),
                    "baseline_wall_p95_s": float(np.quantile([float(baseline[s]["wall_s"]) for s in seeds], 0.95)),
                    "candidate_wall_p95_s": float(np.quantile([float(candidate[s]["wall_s"]) for s in seeds], 0.95)),
                    "mean_distance_delta_m": distance_delta,
                    "mean_measure_delta": float(np.mean([float(candidate[s]["measure_calls"]) - float(baseline[s]["measure_calls"]) for s in seeds])),
                    "mean_switch_delta": float(np.mean([float(candidate[s]["switches"]) - float(baseline[s]["switches"]) for s in seeds])),
                    "mean_clear_delta": float(np.mean([float(candidate[s]["clear_calls"]) - float(baseline[s]["clear_calls"]) for s in seeds])),
                    "mean_failed_clear_delta": float(np.mean([float(candidate[s]["failed_clear"]) - float(baseline[s]["failed_clear"]) for s in seeds])),
                    "mechanism": mechanism,
                    "fallback_counts": {
                        reason: sum(1 for seed in seeds if str(candidate[seed].get("fallback_reason", "")) == reason)
                        for reason in sorted({str(candidate[seed].get("fallback_reason", "")) for seed in seeds})
                    },
                }
                group["mechanism_gate"] = bool(
                    mechanism.get("physical_stations_removed", 0.0) > 0.0
                    or (distance_delta < 0.0 and mean_delta < 0.0)
                )
                groups.append(group)

    summary = {
        "stage": stage,
        "runs": len(rows),
        "max_cost_identity_error_s": max((abs(float(row["cost_identity_error_s"])) for row in rows), default=0.0),
        "groups": groups,
    }
    if stage == "dev":
        selections = []
        for mode in sorted({group["mode"] for group in groups}):
            items = {group["candidate"]: group for group in groups if group["mode"] == mode and group["error_mode"] == "spatial"}
            eligible = []
            checks_by_arm = {}
            for arm, item in items.items():
                checks = {
                    "all_clear": item["all_clear"],
                    "cost_identity": item["cost_identity_ok"],
                    "mean_improves_3pct": item["relative_change_percent"] <= -3.0,
                    "p95_within_103pct": item["candidate_p95"] <= 1.03 * item["baseline_p95"],
                    "max_within_103pct": item["candidate_max"] <= 1.03 * item["baseline_max"],
                    "runtime_p95_within_limit": item["candidate_wall_p95_s"] <= max(2.0 * item["baseline_wall_p95_s"], 10.0),
                    "mechanism_gate": item["mechanism_gate"],
                }
                checks_by_arm[arm] = checks
                if all(checks.values()):
                    eligible.append(arm)
            preferred = min(eligible, key=lambda arm: items[arm]["candidate_mean"], default=None)
            selections.append({
                "mode": mode,
                "eligible": eligible,
                "preferred": preferred,
                "checks": checks_by_arm,
                "code_hash": code_hash(),
                "config_hash": None if preferred is None else _stable_hash({"mode": mode, "arm": preferred, "bearing_decimals": 2}),
            })
        summary["development_selection"] = selections
    else:
        gates = []
        for group in groups:
            if group["error_mode"] != "spatial":
                continue
            stresses = [item for item in groups if item["mode"] == group["mode"] and item["candidate"] == group["candidate"] and item["error_mode"] != "spatial"]
            checks = {
                "all_clear": group["all_clear"] and all(item["all_clear"] for item in stresses),
                "cost_identity": group["cost_identity_ok"] and all(item["cost_identity_ok"] for item in stresses),
                "mean_improves_3pct": group["relative_change_percent"] <= -3.0,
                "ci97_5_upper_below_zero": group["paired_delta_ci97_5"][1] < 0.0,
                "p95_within_103pct": group["candidate_p95"] <= 1.03 * group["baseline_p95"],
                "max_within_103pct": group["candidate_max"] <= 1.03 * group["baseline_max"],
                "stress_means_within_3pct": all(item["relative_change_percent"] <= 3.0 for item in stresses),
                "runtime_p95_within_limit": group["candidate_wall_p95_s"] <= max(2.0 * group["baseline_wall_p95_s"], 10.0),
            }
            gates.append({"mode": group["mode"], "candidate": group["candidate"], "checks": checks, "accepted": all(checks.values())})
        summary["acceptance_gates"] = gates
    return summary


def freeze_manifest(out: Path, stage: str, modes: list[int], arms: list[str]) -> dict:
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "modes": modes,
        "arms": arms,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_head_at_freeze": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_hash": code_hash(),
        "configurations": {f"q{mode}-{arm}": arm_config(mode, arm)
                           for mode in modes for arm in arms},
        "bootstrap_seed": 20260912,
        "seed_ranges": "160000-160049" if stage == "dev" else "280000-280199,281000-281029,281100-281129",
    }
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("resume manifest does not match frozen CVR experiment")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_acceptance_request(summary_path: Path, modes: list[int], arms: list[str], out: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(arms) != {"ISR", next((row["preferred"] for row in summary.get("development_selection", []) if row["mode"] in modes), None)}:
        raise RuntimeError("acceptance arms do not match the frozen development preference")
    for mode in modes:
        row = next((item for item in summary.get("development_selection", []) if item["mode"] == mode), None)
        if row is None or row.get("preferred") is None or row["preferred"] not in arms:
            raise RuntimeError(f"mode {mode} has no eligible frozen development candidate")
        if row.get("code_hash") != code_hash():
            raise RuntimeError("acceptance code hash differs from development selection")
    for manifest in (ROOT / "results").rglob("manifest.json"):
        try:
            if out.resolve() in manifest.resolve().parents:
                continue
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if any(seed in json.dumps(payload) for seed in ("280000", "281000", "281100")):
            raise RuntimeError(f"acceptance seed range already appears in {manifest}")


def _blank_row(mode, seed, kind, error_mode, arm, status) -> dict:
    row = {field: "" for field in FIELDS}
    row.update({"mode": mode, "seed": seed, "kind": kind, "error_mode": error_mode, "arm": arm, "status": status, "code_hash": code_hash(), "config_hash": config_hash(mode, arm)})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "acceptance"], required=True)
    parser.add_argument("--modes", default="3,4")
    parser.add_argument("--arms", default="ISR,COMPAT,CVR")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--development-summary", type=Path)
    args = parser.parse_args()
    modes = [int(value) for value in args.modes.split(",")]
    arms = [value.strip().upper() for value in args.arms.split(",")]
    if not set(arms).issubset({"ISR", "COMPAT", "SEGFIXED", "CVR"}) or "ISR" not in arms:
        raise SystemExit("arms must contain ISR and only ISR,COMPAT,SEGFIXED,CVR")
    out = Path(args.out_dir)
    if args.stage == "acceptance":
        if args.development_summary is None:
            raise SystemExit("acceptance requires --development-summary")
        validate_acceptance_request(args.development_summary, modes, arms, out)
    out.mkdir(parents=True, exist_ok=True)
    freeze_manifest(out, args.stage, modes, arms)

    runs_path = out / "runs.csv"
    existing = []
    if runs_path.exists():
        with runs_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    done = {
        (int(row["mode"]), int(row["seed"]), row["error_mode"], row["arm"])
        for row in existing
        if row.get("status") == "complete"
        and row.get("code_hash") == code_hash()
        and row.get("config_hash") == config_hash(int(row["mode"]), row["arm"])
    }
    decisions_dir = out / "decisions"
    decisions_dir.mkdir(exist_ok=True)
    write_header = not runs_path.exists()
    all_jobs = list(jobs(args.stage, modes, arms))
    with runs_path.open("a", encoding="utf-8-sig" if write_header else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for index, (mode, seed, kind, error_mode, arm) in enumerate(all_jobs, 1):
            key = (mode, seed, error_mode, arm)
            if key in done:
                continue
            writer.writerow(_blank_row(mode, seed, kind, error_mode, arm, "started"))
            handle.flush()
            decision_path = decisions_dir / f"q{mode}_{seed}_{error_mode}_{arm}.jsonl"
            if decision_path.exists():
                attempt = 2
                while decision_path.with_name(decision_path.stem + f".attempt-{attempt}.jsonl").exists():
                    attempt += 1
                decision_path = decision_path.with_name(decision_path.stem + f".attempt-{attempt}.jsonl")
            try:
                row, decisions = run_one(mode, seed, kind, error_mode, arm,
                                         journal_path=decision_path)
            except Exception as exc:
                failure = _blank_row(mode, seed, kind, error_mode, arm, "failed")
                failure.update({"success": False, "error_type": type(exc).__name__, "error": str(exc)})
                writer.writerow(failure)
                handle.flush()
                with (out / "failures.jsonl").open("a", encoding="utf-8") as failures:
                    failures.write(json.dumps(failure, ensure_ascii=False) + "\n")
                raise
            writer.writerow(row)
            handle.flush()
            if not row["success"] or row["sources"] != row["cleared"] or abs(row["cost_identity_error_s"]) > 1e-6:
                raise RuntimeError(f"CVR experiment correctness gate failed: {row}")
            if index % 20 == 0 or index == len(all_jobs):
                print(f"runs={index}/{len(all_jobs)}", flush=True)

    with runs_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize(rows, args.stage)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    from scripts.analyze_cvr_local import write_analysis
    write_analysis(rows, out, args.stage)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
