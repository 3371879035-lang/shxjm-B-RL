"""验证G25OR的P0修复：首次示向楔形严格外包半径1500m的扇形。"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from brl.g25o import initial_track_polygon
from brl.geometry import point_in_convex_polygon

def main():
    rng = np.random.default_rng(20260911)
    bad = 0
    checked = 0
    for _ in range(300000):
        p = rng.uniform(-1880, 1880, 2)
        theta = rng.uniform(0, 360)
        r = 1500.0 * math.sqrt(rng.random())
        ang = math.radians(theta + rng.uniform(-1.01, 1.01))
        z = p + r * np.array([math.cos(ang), math.sin(ang)])
        if np.linalg.norm(z) > 1800.0 + 1e-9:
            continue
        checked += 1
        P = initial_track_polygon(p, theta)
        if not point_in_convex_polygon(z, P):
            bad += 1
    cap = 1500.0 - 1500.0 * math.cos(math.radians(1.01))
    out = {
        "valid_source_points_checked": checked,
        "containment_violations": bad,
        "old_triangle_central_cap_m": cap,
        "new_triangle_outer_radius_m": 1500.0 / math.cos(math.radians(1.01)),
        "epsilon_deg": 1.01,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    (ROOT / "results" / "g25or_p0_verification.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
