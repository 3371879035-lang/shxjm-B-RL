"""新几何方案 G25O 与现有 A0/PPO 底层的严格配对本地比较。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from brl.g25o import G25OPolicy
from brl.local_env import MacroEnv, RadioEnv, Source
from brl.coverage import s25_points, s25_max_triangle_edge


def generate_sources(seed: int, mode: int, kind: str):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 17))
    sources = []
    for ch in rng.choice(np.arange(1, 21), n, replace=False):
        if kind == "edge":
            r = rng.uniform(1650, 1800)
            radius = 1000.0
        else:
            r = 1800.0 * np.sqrt(rng.random())
            radius = rng.uniform(1000, 1500)
        angle = rng.uniform(0, 2 * np.pi)
        pos = r * np.array([np.cos(angle), np.sin(angle)])
        src_kind = "omni"
        direction = 0.0
        if mode == 4 and (kind == "edge" or rng.random() < 0.5):
            src_kind = "directional"
            direction = float(np.degrees(angle)) if kind == "edge" else float(rng.uniform(0, 360))
        sources.append(Source(int(ch), pos, float(radius), src_kind, direction))
    return sources


def run_a0(mode, seed, sources):
    n = len(sources)
    env = MacroEnv(mode=mode, n_sources=n, seed=seed, step_limit=20000)
    env.reset(seed=seed, n_sources=n, sources=sources)
    steps = 0
    while not env.env.done and steps < 1000:
        a = env.heuristic_action()
        env.step(a)
        steps += 1
    return {
        "variant": "A0", "success": bool(env.env.success or env.env.completion_certificate()),
        "cleared": int(env.env.cleared_count()), "sources": n,
        "virtual_time_s": float(env.env.virtual_time), "distance_m": float(env.env.move_distance),
        "measure_calls": int(env.env.n_measure), "switches": int(env.env.n_switch),
        "clear_calls": int(env.env.n_clear), "failed_clear": int(env.env.n_clear_fail),
        "solver_failures": 0, "optical_fallbacks": int(env.env.fallback_actions),
        "extra_measurements": 0, "macro_steps": steps,
    }


def run_g25o(mode, seed, sources):
    n = len(sources)
    env = RadioEnv(mode=mode, n_sources=n, seed=seed, step_limit=20000)
    env.reset(seed=seed, n_sources=n, sources=sources)
    policy = G25OPolicy("G7O" if mode == 3 else "G25O")
    out = policy.run(env)
    out.update({"variant": "G25O", "sources": n})
    out["success"] = bool(out["success"] or env.completion_certificate())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=92000)
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "g25o_paired_validation.csv"))
    args = ap.parse_args()
    rows = []
    t0 = time.time()
    for kind in ("uniform", "edge"):
        for mode in (3, 4):
            for seed in range(args.start, args.start + args.n):
                src = generate_sources(seed, mode, kind)
                a = run_a0(mode, seed, src)
                b = run_g25o(mode, seed, src)
                for method, out in (("A0", a), ("G25O", b)):
                    r = {"kind": kind, "mode": mode, "seed": seed, "method": method}
                    r.update({k: out[k] for k in out if k not in ("variant", "sources")})
                    r["sources"] = len(src)
                    rows.append(r)
            sub = [r for r in rows if r["kind"] == kind and r["mode"] == mode]
            a_rows = [r for r in sub if r["method"] == "A0"]
            b_rows = [r for r in sub if r["method"] == "G25O"]
            print(json.dumps({
                "kind": kind, "mode": mode, "n_pairs": len(a_rows),
                "A0_all_clear": sum(r["cleared"] == r["sources"] for r in a_rows),
                "G25_all_clear": sum(r["cleared"] == r["sources"] for r in b_rows),
                "A0_mean_s": float(np.mean([r["virtual_time_s"] for r in a_rows])),
                "G25_mean_s": float(np.mean([r["virtual_time_s"] for r in b_rows])),
                "A0_dist": float(np.mean([r["distance_m"] for r in a_rows])),
                "G25_dist": float(np.mean([r["distance_m"] for r in b_rows])),
                "G25_fallbacks": int(sum(r["optical_fallbacks"] for r in b_rows)),
                "G25_solver_failures": int(sum(r["solver_failures"] for r in b_rows)),
            }, ensure_ascii=False), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {"n_per_group": args.n, "start_seed": args.start, "wall_s": time.time() - t0,
            "s25_points": len(s25_points()), "s25_max_triangle_edge_m": s25_max_triangle_edge(),
            "rows": len(rows)}
    (ROOT / "results" / "g25o_paired_validation_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
