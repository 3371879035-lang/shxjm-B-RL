"""Plot Clear-as-Search Shadow Audit summary."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "extreme_plan"
OUT = ROOT / "docs" / "extreme_plan" / "clear_shadow_audit_50scenes.png"


def main():
    data = json.loads((RES / "clear_shadow_audit_50scenes_summary.json").read_text(encoding="utf-8"))["summary"]
    r3, r4 = data["problem3"], data["problem4"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=220)
    labels = ["Q3", "Q4"]
    x = np.arange(2); w = 0.25
    axes[0].bar(x - w, [r3["valid_net_s_per_source_mean"], r4["valid_net_s_per_source_mean"]], w,
                label="valid net", color="#c53030")
    axes[0].bar(x, [r3["signal_only_net_s_per_source_mean"], r4["signal_only_net_s_per_source_mean"]], w,
                label="signal-only net", color="#d69e2e")
    axes[0].bar(x + w, [r3["gross_net_s_per_source_mean"], r4["gross_net_s_per_source_mean"]], w,
                label="gross ceiling", color="#2b6cb0")
    axes[0].axhline(30, color="#718096", ls="--", lw=1)
    axes[0].axhline(70, color="#718096", ls=":", lw=1)
    axes[0].text(1.35, 32, "Q3 gate 30-40", fontsize=7)
    axes[0].text(1.35, 72, "Q4 gate 70-100", fontsize=7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("net saving (s/source)")
    axes[0].set_title("(a) Net saving vs gate", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.25)

    names = ["undiscovered true\nat first stop", "shadow signal\ndiscoveries"]
    v3 = [r3["undiscovered_true_at_first_total"], r3["shadow_signal_events_total"]]
    v4 = [r4["undiscovered_true_at_first_total"], r4["shadow_signal_events_total"]]
    xg = np.arange(2); w = 0.35
    axes[1].bar(xg - w / 2, v3, w, label="Q3", color="#2b6cb0")
    axes[1].bar(xg + w / 2, v4, w, label="Q4", color="#c05621")
    for xi, (a, b) in enumerate(zip(v3, v4)):
        axes[1].text(xi - w / 2, a + 8, str(a), ha="center", fontsize=8)
        axes[1].text(xi + w / 2, b + 8, str(b), ha="center", fontsize=8)
    axes[1].set_xticks(xg); axes[1].set_xticklabels(names, fontsize=8)
    axes[1].set_ylabel("count over 50 scenes")
    axes[1].set_title("(b) Discovery substitution", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()