"""官方模拟器入口：G25O几何主动搜索（25点覆盖+双侧区间定位+机会式补测）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brl.coverage import s21_points, s25_points, s3_points
from brl.g25o import G25OPolicy
from brl.remote import OfficialClient, RemoteBelief


def default_robot_id() -> str:
    cfg = ROOT / "config" / "team.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text(encoding="utf-8")).get("robot_id", "")
        except Exception:
            return ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, required=True, choices=[3, 4])
    ap.add_argument("--robot-id", type=str, default=default_robot_id())
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:2026")
    ap.add_argument("--wait-interface-s", type=float, default=1200.0)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--variant", type=str, default="G25O")
    ap.add_argument("--coverage", type=str, default="S25", choices=["S25", "S21", "S4"])
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "results" / f"official_g25o_mode{args.mode}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    req_log = out_dir / "requests.jsonl"
    client = OfficialClient(args.base_url, args.robot_id, log_path=str(req_log))

    wait_start = time.time()
    while True:
        try:
            enter = client.enter()
            if enter.get("accepted") is True:
                break
            raise RuntimeError(f"/enter rejected: {enter}")
        except Exception as e:
            if time.time() - wait_start > args.wait_interface_s:
                raise
            print(f"[wait] interface not ready: {e}", flush=True)
            time.sleep(1.0)

    belief = RemoteBelief(client, mode=args.mode)
    if args.mode == 3:
        cov_points = s3_points()
    elif args.coverage == "S21":
        cov_points = s21_points()
    elif args.coverage == "S4":
        cov_points = __import__("brl.coverage", fromlist=["s4_points"]).s4_points()
    else:
        cov_points = s25_points()
    # 完成证书必须和策略实际扫描的点集一致。
    belief.coverage_points = cov_points
    belief.n_coverage = len(cov_points)
    policy = G25OPolicy(args.variant, coverage=args.coverage)
    print(f"[enter] remaining_real_duration_s={client.remaining_real_duration_s}", flush=True)
    result = policy.run(belief)
    result.update({"mode": args.mode, "variant": args.variant, "coverage": args.coverage,
                   "robot_id": args.robot_id, "request_log": str(req_log)})
    if belief.success and not getattr(belief, "exited", False):
        try:
            exit_resp = client.exit()
            belief.exited = True
            result["exit_response"] = exit_resp
        except Exception as e:
            result["exit_warning"] = str(e)
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
