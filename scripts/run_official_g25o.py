"""官方模拟器入口：G25O几何主动搜索（25点覆盖+双侧区间定位+机会式补测）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brl.coverage import s21_points, s25_points, s3_points, s4_points
from brl.g25o import G25OPolicy
from brl.g21a import G21APolicy
from brl.independent_candidate import IndependentCandidate
from brl.isr_v2 import ISRV2Candidate
from brl.jsp import JSPCandidate
from brl.jsp.model import JSPRanker
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
    ap.add_argument("--planner", choices=["analytic", "learned"], default="analytic")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--coverage", type=str, default="S25", choices=["S25", "S21", "S4"])
    args = ap.parse_args()

    variant = str(args.variant).upper()
    if variant in {"ISR", "ISRV2", "JSP"} and args.coverage != "S25":
        ap.error(f"{variant} fixes Q3 to S3 and Q4 to S25; --coverage must be S25")
    if variant == "JSP" and args.planner == "learned":
        if not args.model:
            ap.error("JSP learned planner requires --model")
        # Validate feature version, problem number, file presence and hash before
        # /enter so a packaging error cannot consume an official attempt.
        try:
            JSPRanker(args.model, args.mode)
        except Exception as exc:
            ap.error(f"invalid JSP model: {exc}")
    if variant != "JSP" and (args.planner != "analytic" or args.model):
        ap.error("--planner/--model are only valid with --variant JSP")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "results" / f"official_g25o_mode{args.mode}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    req_log = out_dir / "requests.jsonl"
    client = OfficialClient(args.base_url, args.robot_id, log_path=str(req_log))
    entered = False
    exit_attempted = False

    result = {"success": False, "mode": args.mode, "variant": args.variant,
              "coverage": args.coverage, "robot_id": args.robot_id,
              "request_log": str(req_log)}
    try:
        wait_start = time.time()
        while True:
            try:
                enter = client.enter()
                if enter.get("accepted") is True:
                    entered = True
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
            cov_points = s4_points()
        else:
            cov_points = s25_points()
        belief.coverage_points = cov_points
        belief.n_coverage = len(cov_points)
        print(f"[enter] remaining_real_duration_s={client.remaining_real_duration_s}", flush=True)
        client.set_action_deadline(reserve_exit_s=30.0)
        if variant == "ISR":
            result.update(IndependentCandidate(mode=args.mode).run(belief))
        elif variant == "ISRV2":
            result.update(ISRV2Candidate(mode=args.mode).run(belief))
        elif variant == "JSP":
            result.update(JSPCandidate(mode=args.mode, planner=args.planner,
                                       model_path=args.model or None).run(belief))
        elif variant.startswith("G21A"):
            result.update(G21APolicy(mode=args.mode).run(belief))
        else:
            result.update(G25OPolicy(args.variant, coverage=args.coverage).run(belief))
        if belief.success and not getattr(belief, "exited", False):
            exit_attempted = True
            exit_resp = client.exit()
            if exit_resp.get("accepted") is not True:
                raise RuntimeError(f"/exit rejected: {exit_resp}")
            belief.exited = True
            result["exit_response"] = exit_resp
    except Exception as exc:
        result.update({"success": False, "error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc()})
        if entered and not exit_attempted:
            exit_attempted = True
            try:
                exit_resp = client.exit()
                result["failure_exit_response"] = exit_resp
                if exit_resp.get("accepted") is not True:
                    result["failure_exit_error"] = f"/exit rejected: {exit_resp}"
            except Exception as exit_exc:
                result["failure_exit_error"] = f"{type(exit_exc).__name__}: {exit_exc}"
        raise
    finally:
        result.update({"mode": args.mode, "variant": args.variant,
                       "coverage": args.coverage, "robot_id": args.robot_id,
                       "request_log": str(req_log)})
        (out_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
