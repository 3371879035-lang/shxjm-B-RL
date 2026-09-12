"""Plot causal decomposition figure from saved summary JSON."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "extreme_plan"
OUT = ROOT / "docs" / "extreme_plan" / "causal_decomposition_50scenes.png"


def main():
    data = json.loads((RES / "causal_decomposition_50scenes_summary.json").read_text(encoding="utf-8"))["summary"]
    r3, r4 = data["problem3"], data["problem4"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=220)
    names = ["G25OR", "C", "B", "A", "O3", "O0"]
    keys = ["baseline_s_per_source", "C_total_s_per_source", "B_total_s_per_source",
            "A_discovery_s_per_source", "O3_s_per_source", "O0_s_per_source"]
    x = np.arange(len(names)); w = 0.36
    axes[0].bar(x - w / 2, [r3[k] for k in keys], w, label="Q3", color="#2b6cb0")
    axes[0].bar(x + w / 2, [r4[k] for k in keys], w, label="Q4", color="#c05621")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylabel("average time per source (s)")
    axes[0].set_title("(a) Strictly paired average time", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.25)
    comps = [("order_gain_G_minus_C", "G-C order loss", "#718096"),
             ("info_cost_C_minus_B", "C-B legal information/proof", "#d69e2e"),
             ("integration_gap_B_minus_O0", "B-O0 search-clear integration", "#e53e3e")]
    bottom = np.zeros(2); xg = np.arange(2)
    for key, name, color in comps:
        vals = np.array([r3["gaps"][key], r4["gaps"][key]])
        axes[1].bar(xg, vals, 0.5, bottom=bottom, label=name, color=color)
        for xi, (b, v) in enumerate(zip(bottom, vals)):
            if v > 4:
                axes[1].text(xi, b + v / 2, f"{v:.1f}", ha="center", va="center", color="white", fontsize=8)
        bottom += vals
    for xi, total in enumerate(bottom):
        axes[1].text(xi, total + 4, f"total {total:.1f}", ha="center", fontsize=9)
    axes[1].set_xticks(xg); axes[1].set_xticklabels(["Q3", "Q4"])
    axes[1].set_ylabel("G25OR - O0 (s/source)")
    axes[1].set_title("(b) Three-term causal decomposition", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()