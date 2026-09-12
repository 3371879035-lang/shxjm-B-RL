"""Plot paired development deltas and cost decomposition for the JSP report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs(root: Path):
    with (root / "runs.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q3", required=True)
    parser.add_argument("--q4", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    roots = {3: Path(args.q3), 4: Path(args.q4)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    colors = {"JSPG": "#3b82f6", "JSPL": "#f97316"}
    for mode, root in roots.items():
        rows = load_runs(root)
        baseline = {int(row["seed"]): row for row in rows if row["arm"] == "ISR"}
        values = []
        labels = []
        for arm in ("JSPG", "JSPL"):
            candidate = {int(row["seed"]): row for row in rows if row["arm"] == arm}
            seeds = sorted(set(baseline) & set(candidate))
            values.append(np.asarray([float(candidate[seed]["seconds_per_source"]) -
                                      float(baseline[seed]["seconds_per_source"]) for seed in seeds]))
            labels.append(arm.replace("JSP", "JSP-"))
        ax = axes[0]
        offset = -0.18 if mode == 3 else 0.18
        positions = np.arange(2) + 1 + offset
        boxes = ax.boxplot(values, positions=positions, widths=.28, patch_artist=True,
                           showfliers=False, medianprops={"color": "black"})
        for box, arm in zip(boxes["boxes"], ("JSPG", "JSPL")):
            box.set_facecolor(colors[arm]); box.set_alpha(.65)
        for x, data in zip(positions, values):
            ax.scatter(np.full(len(data), x), data, s=8, alpha=.22,
                       color="#1f2937", linewidths=0)
        mean_base = np.mean([float(row["seconds_per_source"]) for row in baseline.values()])
        ax.plot(positions, [data.mean() for data in values], "D", color="black", ms=5,
                label="Mean" if mode == 3 else None)
        ax.scatter(positions, [-.03 * mean_base] * 2, marker="_", s=190,
                   color="#15803d", linewidths=2,
                   label="3% improvement gate" if mode == 3 else None)
    axes[0].axhline(0, color="#dc2626", linewidth=1.2)
    axes[0].set_xticks([.82, 1.18, 1.82, 2.18], ["Q3 G", "Q4 G", "Q3 L", "Q4 L"])
    axes[0].set_ylabel("Candidate minus ISR-v1 (seconds/source)")
    axes[0].set_title("Paired whole-episode development deltas")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=.2)

    components = ["movement_seconds_per_source", "measurement_seconds_per_source",
                  "switch_seconds_per_source", "failed_clear_seconds_per_source"]
    names = ["Move", "Measure", "Switch", "Failed clear"]
    entries = []
    for mode, root in roots.items():
        detail = json.loads((root / "cost_decomposition.json").read_text(encoding="utf-8"))
        for item in detail["comparisons"]:
            entries.append((f"Q{mode} {item['candidate'][-1]}",
                            item["component_delta_candidate_minus_isr"]))
    x = np.arange(len(entries)); bottom = np.zeros(len(entries))
    palette = ["#6366f1", "#06b6d4", "#a855f7", "#ef4444"]
    for component, name, color in zip(components, names, palette):
        height = np.asarray([entry[1][component] for entry in entries])
        axes[1].bar(x, height, bottom=bottom, label=name, color=color, alpha=.8)
        bottom += height
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_xticks(x, [entry[0] for entry in entries])
    axes[1].set_ylabel("Mean cost delta (seconds/source)")
    axes[1].set_title("Where the time changed")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.2)
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
