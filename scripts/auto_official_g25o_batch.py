"""全自动官方演练批量运行：UIA点击开始，G25O runner等待接口，完成后返回再开下一局。

只点击问题3/4的开始演练测试按钮，不触碰正式测试。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import win32gui
import win32process
from pywinauto.application import Application

TITLE = "无线电干扰源环境模拟器"
SIM_EXE = r"C:\Users\huani\Desktop\CUMCM2026B\Jammers-simulator-full-win64\Jammers-simulator-full\jammers-simulator-full.exe"


def get_sim_window():
    hwnd = win32gui.FindWindow(None, TITLE)
    if not hwnd:
        raise RuntimeError("simulator window not found")
    win32gui.ShowWindow(hwnd, 9)
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    candidates = []
    try:
        app = Application(backend="uia").connect(process=pid, timeout=2)
        candidates.append(app.window(handle=hwnd))
    except Exception:
        pass
    try:
        app2 = Application(backend="uia").connect(path=SIM_EXE, timeout=2)
        for w in app2.windows():
            try:
                if TITLE in w.window_text():
                    candidates.append(w)
            except Exception:
                pass
    except Exception:
        pass
    for w in candidates:
        try:
            if len(w.descendants(control_type="Button")) > 5:
                return w, pid
        except Exception:
            pass
    if candidates:
        return candidates[0], pid
    raise RuntimeError("simulator UIA window not accessible")


def find_button(w, text, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for b in w.descendants(control_type="Button"):
                try:
                    if b.window_text().strip() == text:
                        return b
                except Exception:
                    continue
        except Exception:
            pass
        time.sleep(0.3)
    return None


def click_button(w, text, timeout=30.0):
    b = find_button(w, text, timeout=timeout)
    if b is None:
        return False
    # 优先走 UIA Invoke Pattern，不依赖窗口是否在前台。
    try:
        b.invoke()
        return True
    except Exception:
        pass
    try:
        b.click_input()
        return True
    except Exception:
        return False


def cleanup_result_dialog(w, mode, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if find_button(w, "返回演练测试", timeout=0.2):
            click_button(w, "返回演练测试", timeout=2)
            time.sleep(0.8)
        if find_button(w, f"开始问题{mode}演练测试", timeout=0.2):
            return True
        if find_button(w, "确认", timeout=0.2):
            click_button(w, "确认", timeout=2)
        time.sleep(0.3)
    return find_button(w, f"开始问题{mode}演练测试", timeout=1) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, required=True, choices=[3, 4])
    ap.add_argument("--runs", type=int, required=True)
    ap.add_argument("--robot-id", type=str, default="202610094088")
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:2026")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--wait-interface-s", type=float, default=180.0)
    ap.add_argument("--variant", type=str, default="G25O")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(1, args.runs + 1):
        run_dir = out_dir / f"run_{i:04d}"
        # 先把上一局的结果弹窗返回主页，再启动 runner。
        try:
            w0, _ = get_sim_window()
            if not find_button(w0, f"开始问题{args.mode}演练测试", timeout=0.5):
                cleanup_result_dialog(w0, args.mode, timeout=20)
        except Exception as e:
            print(f"[batch] run {i}: pre-clean error {e}", flush=True)
        log_out = open(out_dir / f"run_{i:04d}.runner.out.log", "w", encoding="utf-8")
        log_err = open(out_dir / f"run_{i:04d}.runner.err.log", "w", encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / "run_official_g25o.py"),
               "--mode", str(args.mode), "--robot-id", args.robot_id,
               "--base-url", args.base_url, "--wait-interface-s", str(args.wait_interface_s),
               "--variant", args.variant, "--out-dir", str(run_dir)]
        proc = subprocess.Popen(cmd, stdout=log_out, stderr=log_err)
        time.sleep(0.8)
        w, pid = get_sim_window()
        click_button(w, "关闭公告", timeout=0.5)
        start_text = f"开始问题{args.mode}演练测试"
        if not click_button(w, start_text, timeout=30):
            proc.kill()
            print(f"[batch] run {i}: start button not found", flush=True)
            break
        print(f"[batch] run {i}: clicked {start_text}, waiting runner...", flush=True)
        deadline = time.time() + 300
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.5)
        if proc.poll() is None:
            proc.kill()
            print(f"[batch] run {i}: runner timeout", flush=True)
        summary = {}
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append({"run": i, "summary": summary})
        print(f"[batch] run {i}: success={summary.get('success')} cleared={summary.get('cleared')} "
              f"V={summary.get('virtual_time_s')} fallback={summary.get('optical_fallbacks')}", flush=True)
        try:
            w, _ = get_sim_window()
            ok = cleanup_result_dialog(w, args.mode, timeout=30)
            if not ok:
                print(f"[batch] run {i}: result page cleanup failed", flush=True)
                time.sleep(1.0)
        except Exception as e:
            print(f"[batch] run {i}: cleanup error {e}", flush=True)
            time.sleep(1.0)
    (out_dir / "batch_auto_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch] done, {len(results)} runs, summary={out_dir / 'batch_auto_summary.json'}", flush=True)


if __name__ == "__main__":
    main()