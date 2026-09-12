"""Re-execute every archived ISR handoff run and report exact differences."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import platform
import sys
from pathlib import Path

import numpy as np


DATASETS = (
    ("FINAL_holdout200.csv", 220000, 200, "repo"),
    ("FINAL_stress_endpoints30.csv", 230000, 30, "endpoints"),
    ("FINAL_stress_constant30.csv", 240000, 30, "constant"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--handoff-dir",
        default=r"D:\数学建模\incoming\isr_handoff_20260912\B_ISR_independent",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "isr_validation" / "handoff_rerun"),
    )
    args = parser.parse_args()
    root = Path(args.handoff_dir).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    core = importlib.import_module("core")
    variants = importlib.import_module("variants")

    differences = []
    counts = []
    for filename, start, count, error_mode in DATASETS:
        archived_rows = list(csv.DictReader((root / filename).open(encoding="utf-8")))
        archived = {(int(r["mode"]), int(r["seed"]), r["variant"]): r for r in archived_rows}
        mismatch_count = 0
        for mode in (3, 4):
            for offset in range(count):
                seed = start + offset
                kind = "edge" if offset % 3 == 0 else "uniform"
                sources = core.generate_sources(seed, mode, kind)
                for variant in ("base", "final"):
                    env = core.Sim(
                        mode, seed, sources, round_bearing=True, error_mode=error_mode
                    )
                    if variant == "base":
                        policy = core.Baseline(env)
                    else:
                        policy = variants.JointShell(env)
                        policy.max_region_radius = 800.0 if mode == 3 else 300.0
                    policy.run()
                    observed = core.result(env)
                    old = archived[(mode, seed, variant)]
                    delta = {
                        "dataset": filename, "mode": mode, "seed": seed,
                        "kind": kind, "variant": variant,
                        "archived_time": float(old["time"]),
                        "observed_time": float(observed["time"]),
                        "time_delta_s": float(observed["time"] - float(old["time"])),
                        "distance_delta_m": float(observed["distance"] - float(old["distance"])),
                        "measure_delta": int(observed["measures"] - int(old["measures"])),
                        "switch_delta": int(observed["switches"] - int(old["switches"])),
                        "clear_delta": int(observed["cleared"] - int(old["cleared"])),
                        "success_match": str(observed["success"]) == old["success"],
                    }
                    exact = (
                        abs(delta["time_delta_s"]) < 1e-9
                        and abs(delta["distance_delta_m"]) < 1e-9
                        and delta["measure_delta"] == 0
                        and delta["switch_delta"] == 0
                        and delta["clear_delta"] == 0
                        and delta["success_match"]
                    )
                    if not exact:
                        differences.append(delta)
                        mismatch_count += 1
        counts.append({
            "dataset": filename, "expected_runs": len(archived_rows),
            "rerun_runs": 4 * count, "mismatches": mismatch_count,
        })
        print(f"[handoff] {filename}: {4 * count} runs, mismatches={mismatch_count}", flush=True)

    if differences:
        with (out / "differences.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(differences[0]))
            writer.writeheader()
            writer.writerows(differences)
    summary = {
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "datasets": counts, "total_reruns": sum(x["rerun_runs"] for x in counts),
        "total_mismatches": len(differences),
        "fully_reproduced": not differences,
        "known_root_cause": (
            "Symmetric bilateral probes were sorted by raw floating-point travel distance; "
            "platform-level last-bit differences can reverse an exact-distance tie."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
