"""本地 mock HTTP 模拟器：用于在不运行官方模拟器时联调官方适配器。"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from brl.local_env import RadioEnv


class MockState:
    def __init__(self, mode=3, seed=12345, n_sources=12):
        self.env = RadioEnv(mode=mode, n_sources=n_sources, seed=seed, bearing_decimals=2)
        self.entered = False
        self.exited = False

    def reset(self):
        self.env.reset()
        self.entered = True
        self.exited = False
        return {"accepted": True, "real_timestamp_ms": 0, "virtual_time_s": 0.0,
                "max_virtual_duration_s": 360000, "max_real_duration_s": 1200,
                "remaining_real_duration_s": 1200}


STATE = MockState()


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send({"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0}, 400)
            return
        path = self.path
        if path == "/enter":
            self._send(STATE.reset())
        elif path == "/measure":
            if not STATE.entered or STATE.exited:
                self._send({"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0})
                return
            pos = body.get("position", {})
            out = STATE.env.measure([float(pos.get("x", 0)), float(pos.get("y", 0))], int(body.get("channel", 1)))
            resp = {"accepted": True, "real_timestamp_ms": 0, "virtual_time_s": out["virtual_time_s"],
                    "measure_result": out["measure_result"]}
            if out.get("svd_deg") is not None:
                resp["svd_deg"] = out["svd_deg"]
            self._send(resp)
        elif path == "/clear":
            if not STATE.entered or STATE.exited:
                self._send({"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0})
                return
            pos = body.get("position", {})
            out = STATE.env.clear([float(pos.get("x", 0)), float(pos.get("y", 0))], int(body.get("channel", 1)))
            self._send({"accepted": True, "real_timestamp_ms": 0, "virtual_time_s": out["virtual_time_s"],
                        "clear_result": out["clear_result"]})
        elif path == "/exit":
            STATE.exited = True
            self._send({"accepted": True, "real_timestamp_ms": 0,
                        "virtual_time_s": STATE.env.virtual_time, "exit_reason": "user_exit"})
        else:
            self._send({"accepted": False}, 404)

    def do_GET(self):
        self._send({"accepted": False}, 405)


def main(port: int = 2027):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"mock simulator on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2027)
    ap.add_argument("--mode", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-sources", type=int, default=12)
    a = ap.parse_args()
    STATE.env = RadioEnv(mode=a.mode, n_sources=a.n_sources, seed=a.seed, bearing_decimals=2)
    main(a.port)
