"""Collect public-feature counterfactual JSP costs with resumable evidence."""
from __future__ import annotations

import argparse
import copy
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

from brl.jsp import JSPConfig, JSPPolicy
from brl.jsp.snapshot import FEATURE_VERSION, candidate_features
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources


SPLITS = {
    # The originally preregistered ranges were shifted by exactly 10000 after
    # seed 290000 was found in the collector smoke evidence.
    "train": (300000, 256),
    "calibration": (301000, 64),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_manifest(out: Path, modes: list[int], splits: list[str], args,
                    configs: dict[str, JSPConfig]) -> None:
    files = [
        Path(__file__).resolve(), ROOT / "scripts" / "run_g25o_compare.py",
        ROOT / "brl" / "jsp" / "policy.py", ROOT / "brl" / "jsp" / "snapshot.py",
        ROOT / "brl" / "bilateral_state.py", ROOT / "brl" / "bilateral.py",
        ROOT / "brl" / "coverage.py", ROOT / "brl" / "independent_candidate.py",
        ROOT / "brl" / "local_env.py", ROOT / "brl" / "protocol.py",
        ROOT / "brl" / "geometry.py", ROOT / "brl" / "g25o.py",
        ROOT / "brl" / "resolver.py",
    ]
    manifest = {
        "schema_version": 1, "feature_version": FEATURE_VERSION,
        "git_head_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "numpy": np.__version__,
        "modes": modes, "splits": splits,
        "split_seeds": {name: list(SPLITS[name]) for name in splits},
        "edge_rule": "offset_mod_3_eq_0", "bearing_decimals": 2,
        "wall_budget_s": float(args.wall_budget_s),
        "max_scenarios": int(args.max_scenarios),
        "configs": {name: asdict(config) for name, config in configs.items()},
        "seed_audit": ("290000 was occupied by collector smoke; shifted whole ranges by 10000; "
                       "300000-300255 and 301000-301063 were unused before frozen collection"),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
    }
    path = out / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("resume manifest does not match frozen collector")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def environment(mode: int, seed: int, kind: str) -> RadioEnv:
    sources = generate_sources(seed, mode, kind)
    env = RadioEnv(mode=mode, n_sources=len(sources), seed=seed,
                   step_limit=20000, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    return env


def trajectory_snapshots(mode: int, seed: int, kind: str,
                         config: JSPConfig) -> list[JSPPolicy]:
    count = 0
    first = JSPPolicy(environment(mode, seed, kind), config)
    def count_hook(policy, candidates, choice):
        nonlocal count
        count += 1
    first.run(decision_hook=count_hook)
    if count == 0:
        return []
    targets = set(int(x) for x in np.linspace(1, count, min(4, count), dtype=int))
    snapshots: list[JSPPolicy] = []
    second = JSPPolicy(environment(mode, seed, kind), config)
    def capture(policy, candidates, choice):
        if policy.decisions + 1 in targets:
            snapshots.append(copy.deepcopy(policy))
    second.run(decision_hook=capture)
    return snapshots


def branch(snapshot: JSPPolicy, candidate) -> dict:
    policy = copy.deepcopy(snapshot)
    start_time = float(policy.env.virtual_time)
    start_distance = float(policy.env.move_distance)
    start_measure = int(policy.env.n_measure)
    start_switch = int(policy.env.n_switch)
    policy.execute(candidate)
    policy.resume_baseline()
    reconstructed = (policy.env.move_distance / 5 + 5 * policy.env.n_measure
                     + policy.env.n_switch + 3 * policy.env.n_clear_fail
                     + 5 * policy.env.cleared_count())
    return {
        "remaining_time_s": float(policy.env.virtual_time - start_time),
        "remaining_distance_m": float(policy.env.move_distance - start_distance),
        "remaining_measures": int(policy.env.n_measure - start_measure),
        "remaining_switches": int(policy.env.n_switch - start_switch),
        "success": bool(policy.env.completion_certificate()),
        "cleared": int(policy.env.cleared_count()),
        "cost_identity_error_s": float(policy.env.virtual_time - reconstructed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="3,4")
    parser.add_argument("--splits", default="train,calibration")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "jsp" / "counterfactuals_20260913"))
    parser.add_argument("--wall-budget-s", type=float, default=14400.0)
    parser.add_argument("--max-scenarios", type=int, default=0,
                        help="diagnostic cap per mode/split; zero uses the frozen budget")
    args = parser.parse_args()
    modes = [int(x) for x in args.modes.split(",")]
    splits = [x.strip() for x in args.splits.split(",")]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    data_path = out / "branches.jsonl"
    progress_path = out / "progress.json"
    done: set[tuple] = set()
    branch_rows: dict[tuple, dict] = {}
    if data_path.exists():
        for line in data_path.open(encoding="utf-8"):
            row = json.loads(line)
            key = (row["mode"], row["split"], row["seed"], row["trajectory"],
                   row["snapshot_index"], row["candidate_index"])
            done.add(key); branch_rows[key] = row
    label_path = out / "labels.jsonl"
    label_done = set()
    if label_path.exists():
        for line in label_path.open(encoding="utf-8"):
            row = json.loads(line)
            label_done.add((row["mode"], row["split"], row["seed"], row["trajectory"],
                            row["snapshot_index"], row["candidate_index"]))
    started = time.monotonic()
    configs = {
        "JSPBASE": JSPConfig(detour_limit_m=-1.0),
        "JSPG": JSPConfig(),
    }
    freeze_manifest(out, modes, splits, args, configs)
    completed_cases = {key[:4] for key in done}
    with data_path.open("a", encoding="utf-8") as handle, label_path.open("a", encoding="utf-8") as labels:
        for mode in modes:
            for split in splits:
                start_seed, count = SPLITS[split]
                if args.max_scenarios:
                    count = min(count, args.max_scenarios)
                for offset in range(count):
                    seed = start_seed + offset
                    kind = "edge" if offset % 3 == 0 else "uniform"
                    for trajectory, config in configs.items():
                        snapshots = trajectory_snapshots(mode, seed, kind, config)
                        for snapshot_index, snapshot in enumerate(snapshots):
                            candidates = snapshot.candidates()
                            public = snapshot.public_snapshot()
                            results = []
                            for candidate_index, candidate in enumerate(candidates):
                                key = (mode, split, seed, trajectory, snapshot_index, candidate_index)
                                if key in done:
                                    continue
                                result = branch(snapshot, candidate)
                                row = {
                                    "mode": mode, "split": split, "seed": seed, "kind": kind,
                                    "trajectory": trajectory, "snapshot_index": snapshot_index,
                                    "candidate_index": candidate_index, "candidate_kind": candidate.kind,
                                    "candidate_key": candidate.key, "baseline": candidate.baseline,
                                    "feature_version": FEATURE_VERSION,
                                    "features": candidate_features(public, snapshot._public_candidate(candidate)).tolist(),
                                    **result,
                                }
                                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
                                done.add(key); branch_rows[key] = row; results.append(row)
                                if not result["success"] or abs(result["cost_identity_error_s"]) > 1e-6:
                                    raise RuntimeError(f"counterfactual branch failed: {row}")
                            # Labels are written as a separate durable record so
                            # every raw branch remains available for audit.
                            prefix = (mode, split, seed, trajectory, snapshot_index)
                            all_rows = [row for key, row in branch_rows.items() if key[:5] == prefix]
                            base = next((r for r in all_rows if r["baseline"]), None)
                            if base is None:
                                raise RuntimeError("counterfactual snapshot lacks baseline candidate")
                            for row in all_rows:
                                label_key = prefix + (row["candidate_index"],)
                                if label_key in label_done: continue
                                labeled = dict(row)
                                labeled["delta_to_baseline_s"] = row["remaining_time_s"] - base["remaining_time_s"]
                                labels.write(json.dumps(labeled, ensure_ascii=False) + "\n"); labels.flush()
                                label_done.add(label_key)
                        completed_cases.add((mode, split, seed, trajectory))
                        progress_path.write_text(json.dumps({
                            "state": "running", "feature_version": FEATURE_VERSION,
                            "completed_cases": len(completed_cases), "branches": len(done),
                            "last": [mode, split, seed, trajectory],
                            "elapsed_s": time.monotonic() - started,
                        }, ensure_ascii=False, indent=2), encoding="utf-8")
                        if len(completed_cases) % 10 == 0:
                            print(f"cases={len(completed_cases)} branches={len(done)}", flush=True)
                        if time.monotonic() - started >= args.wall_budget_s:
                            progress_path.write_text(json.dumps({
                                "state": "budget_exhausted", "completed_cases": len(completed_cases),
                                "branches": len(done), "elapsed_s": time.monotonic() - started,
                            }, ensure_ascii=False, indent=2), encoding="utf-8")
                            return
    progress_path.write_text(json.dumps({
        "state": "complete", "feature_version": FEATURE_VERSION,
        "completed_cases": len(completed_cases), "branches": len(done),
        "elapsed_s": time.monotonic() - started,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
