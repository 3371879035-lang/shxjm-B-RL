"""Forty-run action-level differential check for the CVR compatibility executor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from brl.cvr import CVRCandidate, CVRConfig
from brl.independent_candidate import IndependentCandidate
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources


def _run(candidate, mode: int, seed: int, kind: str):
    sources = generate_sources(seed, mode, kind)
    env = RadioEnv(mode=mode, n_sources=len(sources), seed=seed,
                   step_limit=20000, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    trace = []
    measure, clear = env.measure, env.clear

    def traced_measure(position, channel, coverage_idx=None, is_refine=False):
        response = measure(position, channel, coverage_idx=coverage_idx, is_refine=is_refine)
        trace.append({
            "kind": "measure", "channel": int(channel),
            "position": list(map(float, position)), "coverage_idx": coverage_idx,
            "is_refine": bool(is_refine), "accepted": response.get("accepted"),
            "result": response.get("measure_result"), "bearing": response.get("svd_deg"),
        })
        return response

    def traced_clear(position, channel):
        response = clear(position, channel)
        trace.append({
            "kind": "clear", "channel": int(channel),
            "position": list(map(float, position)), "coverage_idx": None,
            "is_refine": False, "accepted": response.get("accepted"),
            "result": response.get("clear_result"), "bearing": None,
        })
        return response

    env.measure, env.clear = traced_measure, traced_clear
    result = candidate.run(env)
    return env, result, trace


def _compare(left: list[dict], right: list[dict]):
    if len(left) != len(right):
        return {"reason": "action_count", "baseline": len(left), "compatible": len(right)}
    for index, (a, b) in enumerate(zip(left, right)):
        scalar_keys = ("kind", "channel", "coverage_idx", "is_refine", "accepted", "result")
        if any(a[key] != b[key] for key in scalar_keys):
            return {"reason": "action_fields", "index": index, "baseline": a, "compatible": b}
        if np.linalg.norm(np.asarray(a["position"]) - np.asarray(b["position"])) > 1e-8:
            return {"reason": "position", "index": index, "baseline": a, "compatible": b}
        av, bv = a["bearing"], b["bearing"]
        if av is None or bv is None:
            if av != bv:
                return {"reason": "bearing_presence", "index": index, "baseline": a, "compatible": b}
        elif abs(float(av) - float(bv)) > 1e-12:
            return {"reason": "bearing", "index": index, "baseline": a, "compatible": b}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for mode in (3, 4):
        for offset in range(10):
            seed = 160000 + offset
            kind = "edge" if offset % 3 == 0 else "uniform"
            base_env, base_result, base_trace = _run(IndependentCandidate(mode), mode, seed, kind)
            compat_env, compat_result, compat_trace = _run(
                CVRCandidate(mode, CVRConfig(planner_enabled=False)), mode, seed, kind)
            difference = _compare(base_trace, compat_trace)
            rows.append({
                "mode": mode, "seed": seed, "kind": kind,
                "actions": len(base_trace), "equivalent": difference is None,
                "first_difference": difference,
                "baseline_time_s": base_env.virtual_time,
                "compatible_time_s": compat_env.virtual_time,
                "time_delta_s": compat_env.virtual_time - base_env.virtual_time,
                "baseline_cleared": base_env.cleared_count(),
                "compatible_cleared": compat_env.cleared_count(),
                "baseline_success": bool(base_result["success"]),
                "compatible_success": bool(compat_result["success"]),
            })
    result = {
        "schema_version": 1, "coordinate_tolerance_m": 1e-8,
        "cost_tolerance_s": 1e-6, "scenario_pairs": len(rows),
        "policy_runs": 2 * len(rows),
        "all_equivalent": all(row["equivalent"] and abs(row["time_delta_s"]) <= 1e-6
                              for row in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["all_equivalent"] else 1)


if __name__ == "__main__":
    main()
