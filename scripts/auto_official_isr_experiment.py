"""Resumable, practice-only multi-variant experiment driver.

The driver first runs 12 smoke practices (three per problem/variant).  It enters
the 400-run experiment only if every smoke run is fully verified.  It never
searches for or clicks a formal-test button.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import msvcrt
import os
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auto_official_g25o_batch import find_button, get_sim_window

DEFAULT_DB = Path(
    r"C:\Users\huani\Desktop\CUMCM2026B\Jammers-simulator-full-win64\Jammers-simulator-full"
    r"\JammersSimulatorData\practice-statistics-queue.sqlite3"
)
RUNNER = ROOT / "scripts" / "run_official_g25o.py"
PRACTICE_BUTTONS = {3: "开始问题3演练测试", 4: "开始问题4演练测试"}


class SingleInstance:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if self.handle.tell() == 0 and self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(f"another ISR official experiment holds {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def assert_no_other_runner() -> None:
    command = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match '^python' -and $_.CommandLine -match 'run_official_g25o.py' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    found = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        text=True, capture_output=True, timeout=15,
    )
    if found.returncode != 0:
        raise RuntimeError(f"cannot inspect existing runners: {found.stderr.strip()}")
    if found.stdout.strip():
        raise RuntimeError(f"an official runner is already active: {found.stdout.strip()}")


def db_connect(path: Path):
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def db_max_id(path: Path) -> int:
    with db_connect(path) as connection:
        row = connection.execute("SELECT COALESCE(MAX(id),0) AS id FROM practice_statistics_tasks").fetchone()
    return int(row["id"])


def match_database_record(path: Path, after_id: int, mode: int, robot_id: str, timeout_s: float = 30.0):
    deadline = time.time() + timeout_s
    rows = []
    while time.time() < deadline:
        with db_connect(path) as connection:
            rows = connection.execute(
                """SELECT id,team_no,problem_no,practice_run_no,case_code,entered,end_reason,
                          cleared_jammer_count,measure_accepted_count,virtual_time_us,
                          program_run_duration_ms,channel_switch_count,clear_failure_count,
                          jammer_count,state,created_at_ms,updated_at_ms
                     FROM practice_statistics_tasks
                    WHERE id>? AND problem_no=? AND team_no=? ORDER BY id""",
                (after_id, mode, str(robot_id)),
            ).fetchall()
        if rows:
            break
        time.sleep(0.5)
    return [dict(row) for row in rows]


def invoke_exact(window, text: str, timeout: float) -> bool:
    button = find_button(window, text, timeout=timeout)
    if button is None:
        return False
    try:
        button.invoke()
        return True
    except Exception:
        return False


def return_to_practice(mode: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        window, _ = get_sim_window()
        if find_button(window, PRACTICE_BUTTONS[mode], timeout=0.2):
            return True
        if find_button(window, "返回演练测试", timeout=0.2):
            if not invoke_exact(window, "返回演练测试", 1.0):
                return False
            time.sleep(0.8)
            continue
        if find_button(window, "确认", timeout=0.2):
            if not invoke_exact(window, "确认", 1.0):
                return False
        time.sleep(0.3)
    return False


def make_schedule(smoke_runs: int, main_runs: int, seed: int,
                  variants: list[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    variants = variants or ["G25OR", "ISR"]
    schedule = []
    global_index = 0
    for phase, blocks in (("smoke", smoke_runs), ("main", main_runs)):
        for block in range(1, blocks + 1):
            jobs = [(mode, variant) for mode in (3, 4) for variant in variants]
            rng.shuffle(jobs)
            for position, (mode, variant) in enumerate(jobs, 1):
                global_index += 1
                schedule.append({
                    "run_id": f"{phase}-b{block:03d}-p{position}-{variant.lower()}-q{mode}",
                    "global_index": global_index, "phase": phase, "block": block,
                    "block_position": position, "mode": mode, "variant": variant,
                    "coverage": "S25",
                })
    return schedule


def load_or_create_manifest(out: Path, args) -> dict:
    path = out / "manifest.json"
    code_files = [
        RUNNER, ROOT / "brl" / "independent_candidate.py", ROOT / "brl" / "bilateral.py",
        ROOT / "brl" / "protocol.py", ROOT / "brl" / "remote.py",
        ROOT / "brl" / "isr_v2.py", Path(__file__).resolve(),
    ]
    generated = {
        "schema_version": 1, "practice_only": True,
        "schedule_seed": args.schedule_seed,
        "smoke_runs_per_group": args.smoke_runs_per_group,
        "main_runs_per_group": args.main_runs_per_group,
        "variants": args.variants,
        "robot_id": str(args.robot_id), "base_url": args.base_url,
        "database": str(Path(args.database).resolve()),
        "runner_timeout_s": float(args.runner_timeout_s),
        "python_executable": str(Path(sys.executable).resolve()),
        "git_head": git_head(),
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in code_files},
        "schedule": make_schedule(
            args.smoke_runs_per_group, args.main_runs_per_group, args.schedule_seed,
            args.variants,
        ),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "schedule_seed", "smoke_runs_per_group", "main_runs_per_group",
            "robot_id", "base_url", "database", "runner_timeout_s",
            "python_executable", "git_head", "code_sha256",
        ):
            if existing.get(key) != generated.get(key):
                raise RuntimeError(f"resume manifest mismatch for {key}")
        if existing.get("variants", ["G25OR", "ISR"]) != generated["variants"]:
            raise RuntimeError("resume manifest mismatch for variants")
        return existing
    atomic_json(path, generated)
    return generated


def write_ledger(out: Path, attempts: list[dict]) -> None:
    if not attempts:
        return
    fields = [
        "run_id", "phase", "block", "block_position", "mode", "variant", "status",
        "technical_failure", "confirmed_not_clear", "runner_returncode", "db_match_status",
        "db_id", "practice_run_no", "case_code", "end_reason", "jammer_count",
        "cleared_jammer_count", "virtual_time_s", "program_run_duration_ms",
        "measure_accepted_count", "channel_switch_count", "clear_failure_count", "run_dir",
    ]
    with (out / "ledger.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(attempts)


def run_job(out: Path, job: dict, args) -> dict:
    run_dir = out / "runs" / job["run_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    before_id = db_max_id(Path(args.database))
    attempt = {**job, "status": "prepared", "run_dir": str(run_dir),
               "db_before_id": before_id, "prepared_at_ms": int(time.time() * 1000)}
    atomic_json(run_dir / "attempt.json", attempt)

    if not return_to_practice(job["mode"], timeout=30):
        attempt.update(status="technical_failure", technical_failure=True,
                       error="cannot reach the requested practice start page")
        atomic_json(run_dir / "attempt.json", attempt)
        return attempt

    out_log = (run_dir / "runner.out.log").open("w", encoding="utf-8")
    err_log = (run_dir / "runner.err.log").open("w", encoding="utf-8")
    command = [
        sys.executable, str(RUNNER), "--mode", str(job["mode"]),
        "--robot-id", str(args.robot_id), "--base-url", args.base_url,
        "--wait-interface-s", "180", "--variant", job["variant"],
        "--coverage", "S25", "--out-dir", str(run_dir / "runner"),
    ]
    process = subprocess.Popen(command, cwd=ROOT, stdout=out_log, stderr=err_log)
    started = False
    for _ in range(2):
        time.sleep(0.8)
        window, _ = get_sim_window()
        invoke_exact(window, "关闭公告", 0.2)
        if invoke_exact(window, PRACTICE_BUTTONS[job["mode"]], 20):
            started = True
            break
        return_to_practice(job["mode"], timeout=10)
    if not started:
        process.kill()
        process.wait(timeout=10)
        out_log.close(); err_log.close()
        attempt.update(status="technical_failure", technical_failure=True,
                       error="practice start button could not be invoked")
        atomic_json(run_dir / "attempt.json", attempt)
        return attempt

    attempt.update(status="started", scenario_started=True, started_at_ms=int(time.time() * 1000))
    atomic_json(run_dir / "attempt.json", attempt)
    try:
        returncode = process.wait(timeout=args.runner_timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        returncode = -9
        attempt["timeout"] = True
    finally:
        out_log.close(); err_log.close()

    summary_path = run_dir / "runner" / "summary.json"
    runner_summary = {}
    if summary_path.exists():
        try:
            runner_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            attempt["summary_error"] = str(exc)
    rows = match_database_record(
        Path(args.database), before_id, job["mode"], str(args.robot_id), timeout_s=30
    )
    if len(rows) == 1:
        db_record = rows[0]
        match_status = "verified"
    elif not rows:
        db_record = {}
        match_status = "missing"
    else:
        db_record = {}
        match_status = "ambiguous"
        attempt["database_candidates"] = rows

    attempt.update({
        "runner_returncode": returncode, "runner_summary": runner_summary,
        "db_match_status": match_status, "technical_failure": False,
        "confirmed_not_clear": False, "finished_at_ms": int(time.time() * 1000),
    })
    if db_record:
        attempt.update({
            "db_id": db_record["id"], "practice_run_no": db_record["practice_run_no"],
            "case_code": db_record["case_code"], "end_reason": db_record["end_reason"],
            "jammer_count": db_record["jammer_count"],
            "cleared_jammer_count": db_record["cleared_jammer_count"],
            "virtual_time_s": db_record["virtual_time_us"] / 1_000_000.0,
            "program_run_duration_ms": db_record["program_run_duration_ms"],
            "measure_accepted_count": db_record["measure_accepted_count"],
            "channel_switch_count": db_record["channel_switch_count"],
            "clear_failure_count": db_record["clear_failure_count"],
            "database_state": db_record["state"],
        })
        attempt["confirmed_not_clear"] = (
            int(db_record["cleared_jammer_count"]) != int(db_record["jammer_count"])
        )
    if returncode != 0 or not runner_summary or match_status != "verified":
        attempt["technical_failure"] = True
    if runner_summary and not runner_summary.get("success", False):
        attempt["technical_failure"] = True
    if attempt["confirmed_not_clear"]:
        attempt["status"] = "confirmed_not_clear"
    elif attempt["technical_failure"]:
        attempt["status"] = "technical_failure"
    else:
        attempt["status"] = "verified_full_clear"
    atomic_json(run_dir / "attempt.json", attempt)
    try:
        return_to_practice(job["mode"], timeout=30)
    except Exception as exc:
        attempt["cleanup_warning"] = str(exc)
        atomic_json(run_dir / "attempt.json", attempt)
    return attempt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--robot-id", default="202610094088")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--database", default=str(DEFAULT_DB))
    parser.add_argument("--smoke-runs-per-group", type=int, default=3)
    parser.add_argument("--main-runs-per-group", type=int, default=100)
    parser.add_argument("--variants", default="G25OR,ISR",
                        help="comma-separated runner variants, e.g. ISR,ISRV2")
    parser.add_argument("--schedule-seed", type=int, default=20260912)
    parser.add_argument("--runner-timeout-s", type=float, default=1170.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.variants = [value.strip().upper() for value in args.variants.split(",") if value.strip()]
    if not args.variants or any(value not in {"G25OR", "ISR", "ISRV2"} for value in args.variants):
        raise SystemExit("--variants must contain G25OR, ISR, or ISRV2")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    with SingleInstance(out / ".official_experiment.lock"):
        assert_no_other_runner()
        manifest = load_or_create_manifest(out, args)
        progress_path = out / "progress.json"
        if progress_path.exists():
            if not args.resume:
                raise RuntimeError("progress already exists; pass --resume to continue the frozen schedule")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        else:
            progress = {"schema_version": 1, "state": "ready", "attempts": [],
                        "next_global_index": 1, "consecutive_technical_failures": 0}
            atomic_json(progress_path, progress)

        interrupted = progress.pop("active_run_id", None)
        if interrupted and not any(a["run_id"] == interrupted for a in progress["attempts"]):
            attempt_path = out / "runs" / interrupted / "attempt.json"
            recovered = json.loads(attempt_path.read_text(encoding="utf-8")) if attempt_path.exists() else {
                "run_id": interrupted, "phase": "unknown", "block": -1, "block_position": -1,
                "mode": -1, "variant": "unknown", "run_dir": str(out / "runs" / interrupted),
            }
            recovered.update(status="technical_failure", technical_failure=True,
                             confirmed_not_clear=False, error="batch interrupted before final accounting")
            progress["attempts"].append(recovered)
            progress["consecutive_technical_failures"] = progress.get("consecutive_technical_failures", 0) + 1
            atomic_json(progress_path, progress)

        if args.prepare_only:
            print(json.dumps({"state": progress["state"], "schedule": len(manifest["schedule"]),
                              "out_dir": str(out)}, ensure_ascii=False), flush=True)
            return

        completed_ids = {a["run_id"] for a in progress["attempts"]}
        for job in manifest["schedule"]:
            if job["run_id"] in completed_ids:
                continue
            if job["phase"] == "main":
                smoke = [a for a in progress["attempts"] if a["phase"] == "smoke"]
                expected = args.smoke_runs_per_group * 2 * len(args.variants)
                if len(smoke) != expected or any(a["status"] != "verified_full_clear" for a in smoke):
                    progress.update(state="stopped", stop_reason="smoke_gate_failed")
                    atomic_json(progress_path, progress)
                    break
            progress.update(state="running", active_run_id=job["run_id"])
            atomic_json(progress_path, progress)
            attempt = run_job(out, job, args)
            progress["attempts"].append(attempt)
            completed_ids.add(job["run_id"])
            progress["next_global_index"] = job["global_index"] + 1
            progress.pop("active_run_id", None)
            if attempt["technical_failure"]:
                progress["consecutive_technical_failures"] += 1
            else:
                progress["consecutive_technical_failures"] = 0
            write_ledger(out, progress["attempts"])
            if attempt["confirmed_not_clear"]:
                progress.update(state="stopped", stop_reason="confirmed_algorithm_not_clear")
                atomic_json(progress_path, progress)
                break
            if progress["consecutive_technical_failures"] >= 2:
                progress.update(state="stopped", stop_reason="two_consecutive_technical_failures")
                atomic_json(progress_path, progress)
                break
            atomic_json(progress_path, progress)
        else:
            progress.update(state="complete", completed_at_ms=int(time.time() * 1000))
            atomic_json(progress_path, progress)
        print(json.dumps({
            "state": progress["state"], "attempts": len(progress["attempts"]),
            "stop_reason": progress.get("stop_reason"), "out_dir": str(out),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
