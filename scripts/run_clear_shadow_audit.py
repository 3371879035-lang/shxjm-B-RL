"""Clear-as-Search Shadow Audit: replay G25OR traces, test whether clear/localization stops
could have replaced future S3/S25 search tasks.  Reuses the existing 50+50 paired scenes."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from brl.local_env import RadioEnv
from brl.g25o import G25OPolicy
from brl.certificates import grid_cells_intersecting_disk, q3_absent_cells, q4_absent_cells
from scripts.run_g25o_compare import generate_sources


def make_env(mode, seed, sources):
    env = RadioEnv(mode=mode, n_sources=len(sources), seed=seed, step_limit=20000)
    env.reset(seed=seed, n_sources=len(sources), sources=sources)
    return env


def shadow_outcome(env, q, ch):
    src = env.source_by_channel.get(int(ch))
    if src is None:
        return "no_signal"
    q = np.asarray(q, dtype=float)
    d = float(np.linalg.norm(q - src.position))
    if d > src.radius + 1e-9:
        return "no_signal"
    if src.kind == "directional":
        n = np.array([np.cos(np.radians(src.direction_deg)), np.sin(np.radians(src.direction_deg))])
        if float(np.dot(n, q - src.position)) < -1e-9:
            return "no_signal"
    return "near" if d <= 5.0 + 1e-9 else "direction"


def trace_g25or(mode, seed, kind):
    sources = generate_sources(seed, mode, kind)
    env = make_env(mode, seed, sources)
    actions = []
    orig_measure = env.measure
    orig_clear = env.clear

    def measure(position, channel, coverage_idx=None, is_refine=False):
        before = {"cur": int(env.current_channel),
                  "unknown": [c for c in range(1, 21) if env.channels[c].status == "unknown"]}
        out = orig_measure(position, channel, coverage_idx=coverage_idx, is_refine=is_refine)
        actions.append({"i": len(actions), "kind": "measure", "p": np.asarray(position, dtype=float).copy(),
                        "ch": int(channel), "coverage_idx": coverage_idx, "is_refine": bool(is_refine),
                        "result": out.get("measure_result"), "before": before})
        return out

    def clear(position, channel):
        before = {"cur": int(env.current_channel),
                  "unknown": [c for c in range(1, 21) if env.channels[c].status == "unknown"]}
        out = orig_clear(position, channel)
        actions.append({"i": len(actions), "kind": "clear", "p": np.asarray(position, dtype=float).copy(),
                        "ch": int(channel), "coverage_idx": None, "is_refine": False,
                        "result": out.get("clear_result"), "before": before})
        return out

    env.measure = measure
    env.clear = clear
    result = G25OPolicy(variant="G25OR", coverage="S25").run(env)
    return sources, actions, env, result


def is_coverage(a):
    return a["kind"] == "measure" and a.get("coverage_idx") is not None and not a["is_refine"]


def replay(actions, deleted, inserted):
    pos = np.zeros(2, dtype=float)
    cur = 1
    total_t = 0.0
    total_d = 0.0
    for i, a in enumerate(actions):
        if i in deleted:
            continue
        q = a["p"]
        move = float(np.linalg.norm(q - pos))
        total_d += move
        pos = q.copy()
        if a["kind"] == "measure":
            ch = int(a["ch"])
            total_t += move / 5.0 + (1.0 if ch != cur else 0.0) + 5.0
            cur = ch
        else:
            total_t += move / 5.0 + (5.0 if a["result"] == "success" else 3.0)
        for ch in inserted.get(i, []):
            total_t += 5.0 + (1.0 if int(ch) != cur else 0.0)
            cur = int(ch)
    return total_t, total_d


def coverage_visits(actions):
    visits = []
    for i, a in enumerate(actions):
        if not is_coverage(a):
            continue
        if visits and visits[-1]["idx"] == int(a["coverage_idx"]):
            visits[-1]["actions"].append(i)
        else:
            visits.append({"idx": int(a["coverage_idx"]), "actions": [i], "p": a["p"].copy()})
    return visits


def audit_scene(mode, seed, kind, gap_per_source):
    sources, actions, env, result = trace_g25or(mode, seed, kind)
    n = max(len(sources), 1)
    noncov = [i for i, a in enumerate(actions) if not is_coverage(a)]
    if not noncov:
        return None
    first = noncov[0]
    actual_channels = {int(s.channel) for s in sources}
    undiscovered_true_at_first = len(set(actions[first]["before"]["unknown"]) & actual_channels)
    cov_indices = [i for i, a in enumerate(actions) if is_coverage(a)]
    # unique natural non-coverage stops: first occurrence of each position
    seen = set()
    stops = []
    for i in noncov:
        p = actions[i]["p"]
        key = (round(float(p[0]), 6), round(float(p[1]), 6))
        if key in seen:
            continue
        seen.add(key)
        stops.append((i, actions[i]))
    # original negative points available before the first stop
    prefix_neg = {c: [] for c in range(1, 21)}
    for i, a in enumerate(actions):
        if i > first:
            break
        if a["kind"] == "measure" and a.get("result") == "no_signal":
            prefix_neg[int(a["ch"])].append(a["p"])
    shadow_neg = {c: [] for c in range(1, 21)}
    shadow_signal_events = []
    shadow_found = set()
    inserted = {}
    total_unknown_opportunities = 0
    for i, a in stops:
        step_channels = []
        for ch in a["before"]["unknown"]:
            if ch in shadow_found:
                continue
            total_unknown_opportunities += 1
            out = shadow_outcome(env, a["p"], ch)
            step_channels.append(int(ch))
            if out == "no_signal":
                shadow_neg[int(ch)].append(a["p"])
            else:
                shadow_found.add(int(ch))
                shadow_signal_events.append((i, int(ch)))
        if step_channels:
            inserted[i] = step_channels
    # free certificate upper bound: does any channel become globally absent using
    # prefix negatives + all shadow no_signal points?  (later original negatives excluded)
    cells, half = grid_cells_intersecting_disk(200.0, 1800.0)
    global_absent = []
    for ch in range(1, 21):
        neg = np.asarray(prefix_neg[ch] + shadow_neg[ch], dtype=float)
        if mode == 3:
            absent_all = bool(len(neg) > 0 and q3_absent_cells(neg, cells, half).all())
        else:
            absent_all = bool(len(neg) >= 3 and q4_absent_cells(neg, cells, half).all())
        if absent_all:
            global_absent.append(ch)
    # valid online deletion: shadow-discovered channels remove their future coverage tasks;
    # plus global-absent channels (free version) as an upper variant.
    signal_deleted = set()
    for i, ch in shadow_signal_events:
        for j in cov_indices:
            if j > i and int(actions[j]["ch"]) == ch:
                signal_deleted.add(j)
    absent_deleted = set()
    for ch in global_absent:
        for j in cov_indices:
            # conservative: delete only tasks after the last shadow stop for this channel
            if j > stops[-1][0] and int(actions[j]["ch"]) == ch:
                absent_deleted.add(j)
    valid_deleted = signal_deleted | absent_deleted
    # shadow-all measurement cost and new plan (only signal+absent task deletion)
    valid_t, valid_d = replay(actions, valid_deleted, inserted)
    base_t_without_insert, _ = replay(actions, valid_deleted, {})
    added_shadow_s = valid_t - base_t_without_insert
    # signal-only oracle variant: only measure channels that would signal
    inserted_signal = {}
    for i, a in stops:
        chs = [ch for ch in a["before"]["unknown"] if ch in shadow_found]
        if chs:
            inserted_signal[i] = chs
    signal_t, signal_d = replay(actions, valid_deleted, inserted_signal)
    # gross upper bound: delete every future coverage task after the first non-coverage stop
    gross_deleted = {j for j in cov_indices if j > first}
    gross_t, gross_d = replay(actions, gross_deleted, {})
    visits = coverage_visits(actions)
    future_visits = []
    for v in visits:
        fv = [i for i in v["actions"] if i > first]
        if fv:
            future_visits.append(fv)
    whole_valid = sum(1 for fv in future_visits if all(i in valid_deleted for i in fv))
    partial_valid = sum(1 for fv in future_visits if any(i in valid_deleted for i in fv) and not all(i in valid_deleted for i in fv))
    whole_gross = len(future_visits)
    return {
        "mode": mode, "seed": seed, "sources": n,
        "orig_v": float(result["virtual_time_s"]), "orig_dist": float(result["distance_m"]),
        "first_noncoverage_index": int(first), "undiscovered_true_at_first": int(undiscovered_true_at_first),
        "noncoverage_stops": len(stops),
        "coverage_tasks_after_first": int(sum(1 for j in cov_indices if j > first)),
        "coverage_visits_after_first": len(future_visits),
        "unknown_opportunities": int(total_unknown_opportunities),
        "shadow_signal_events": int(len(shadow_signal_events)),
        "shadow_absent_channels": int(len(global_absent)),
        "valid_deleted_tasks": int(len(valid_deleted)),
        "valid_whole_waypoints": int(whole_valid), "valid_partial_waypoints": int(partial_valid),
        "valid_dist_saved": float(result["distance_m"] - valid_d),
        "added_shadow_s": float(added_shadow_s),
        "valid_new_v": float(valid_t), "valid_net_s_per_source": float((result["virtual_time_s"] - valid_t) / n),
        "valid_recovery_of_B_minus_O0": float((result["virtual_time_s"] - valid_t) / n / gap_per_source) if gap_per_source and gap_per_source > 1e-9 else None,
        "signal_only_added_s": float(signal_t - base_t_without_insert),
        "signal_only_net_s_per_source": float((result["virtual_time_s"] - signal_t) / n),
        "gross_deleted_tasks": int(len(gross_deleted)),
        "gross_whole_waypoints": int(whole_gross),
        "gross_dist_saved": float(result["distance_m"] - gross_d),
        "gross_new_v": float(gross_t),
        "gross_net_s_per_source": float((result["virtual_time_s"] - gross_t) / n),
        "gross_recovery_of_B_minus_O0": float((result["virtual_time_s"] - gross_t) / n / gap_per_source) if gap_per_source and gap_per_source > 1e-9 else None,
    }


def load_gaps():
    p = ROOT / "results" / "extreme_plan" / "causal_decomposition_50scenes.csv"
    gaps = {}
    if p.exists():
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                gaps[(int(r["mode"]), int(r["seed"]))] = float(r["integration_gap_per_source"])
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=160000)
    args = ap.parse_args()
    gaps = load_gaps()
    rows = []
    for mode in (3, 4):
        for i in range(args.n):
            seed = args.start + i
            kind = "edge" if i % 3 == 0 else "uniform"
            gap = gaps.get((mode, seed), float("nan"))
            row = audit_scene(mode, seed, kind, gap)
            if row is not None:
                rows.append(row)
        print("[shadow-audit] mode", mode, "done", flush=True)
    out_csv = ROOT / "results" / "extreme_plan" / "clear_shadow_audit_50scenes.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {}
    for mode in (3, 4):
        sub = [r for r in rows if r["mode"] == mode]
        def mean(k):
            vals = [r[k] for r in sub if r[k] is not None]
            return float(np.mean(vals)) if vals else None
        def median(k):
            vals = [r[k] for r in sub if r[k] is not None]
            return float(np.median(vals)) if vals else None
        s = {
            "n_scenes": len(sub), "sources_total": int(sum(r["sources"] for r in sub)),
            "first_noncoverage_index_mean": mean("first_noncoverage_index"), "first_noncoverage_index_median": median("first_noncoverage_index"),
            "undiscovered_true_at_first_mean": mean("undiscovered_true_at_first"),
            "undiscovered_true_at_first_total": int(sum(r["undiscovered_true_at_first"] for r in sub)),
            "scenes_all_true_discovered_before_first": int(sum(1 for r in sub if r["undiscovered_true_at_first"] == 0)),
            "noncoverage_stops_mean": mean("noncoverage_stops"),
            "coverage_tasks_after_first_mean": mean("coverage_tasks_after_first"),
            "coverage_visits_after_first_mean": mean("coverage_visits_after_first"),
            "unknown_opportunities_mean": mean("unknown_opportunities"),
            "shadow_signal_events_total": int(sum(r["shadow_signal_events"] for r in sub)),
            "shadow_signal_scenes": int(sum(1 for r in sub if r["shadow_signal_events"] > 0)),
            "shadow_absent_channels_total": int(sum(r["shadow_absent_channels"] for r in sub)),
            "valid_deleted_tasks_mean": mean("valid_deleted_tasks"),
            "valid_whole_waypoints_mean": mean("valid_whole_waypoints"),
            "valid_partial_waypoints_mean": mean("valid_partial_waypoints"),
            "valid_dist_saved_mean": mean("valid_dist_saved"),
            "added_shadow_s_mean": mean("added_shadow_s"),
            "valid_net_s_per_source_mean": mean("valid_net_s_per_source"),
            "valid_net_s_per_source_median": median("valid_net_s_per_source"),
            "valid_net_s_per_source_max": float(max(r["valid_net_s_per_source"] for r in sub)),
            "valid_recovery_of_B_minus_O0_mean": mean("valid_recovery_of_B_minus_O0"),
            "signal_only_added_s_mean": mean("signal_only_added_s"),
            "signal_only_net_s_per_source_mean": mean("signal_only_net_s_per_source"),
            "gross_deleted_tasks_mean": mean("gross_deleted_tasks"),
            "gross_whole_waypoints_mean": mean("gross_whole_waypoints"),
            "gross_dist_saved_mean": mean("gross_dist_saved"),
            "gross_net_s_per_source_mean": mean("gross_net_s_per_source"),
            "gross_net_s_per_source_median": median("gross_net_s_per_source"),
            "gross_net_s_per_source_max": float(max(r["gross_net_s_per_source"] for r in sub)),
            "gross_recovery_of_B_minus_O0_mean": mean("gross_recovery_of_B_minus_O0"),
        }
        if mode == 3:
            s["gate_net_s_per_source"] = "30-40"
            s["gate_pass"] = bool(s["valid_net_s_per_source_mean"] is not None and s["valid_net_s_per_source_mean"] >= 30.0)
        else:
            s["gate_net_s_per_source"] = "70-100"
            s["gate_pass"] = bool(s["valid_net_s_per_source_mean"] is not None and s["valid_net_s_per_source_mean"] >= 70.0)
        s["gross_ceiling_can_pass"] = bool((s["gross_net_s_per_source_mean"] or 0) >= (30.0 if mode == 3 else 70.0))
        summary[f"problem{mode}"] = s
    out = {"n": args.n, "start": args.start, "summary": summary}
    (ROOT / "results" / "extreme_plan" / "clear_shadow_audit_50scenes_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])


if __name__ == "__main__":
    main()