"""从 OfficialClient 的 JSONL 请求日志统计官方演练指标。"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def load(path):
    rows=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            o=json.loads(line)
        except Exception:
            continue
        if o.get("response",{}).get("accepted") is True:
            rows.append(o)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="")
    args=ap.parse_args()
    rows=load(args.log)
    pos=np.array([0.0,0.0])
    move=0.0
    measure=0; clear=0; switch=0; fail_clear=0
    current_channel=1
    measure_results=Counter(); clear_results=Counter()
    cleared_channels=set()
    vtime=0.0
    real_times=[]
    for r in rows:
        path=r.get("path")
        p=np.array([r["payload"]["position"]["x"], r["payload"]["position"]["y"]],dtype=float) if "position" in r.get("payload",{}) else None
        if p is not None:
            move += float(np.linalg.norm(p-pos))
            pos=p
        resp=r.get("response",{})
        if "virtual_time_s" in resp:
            vtime=float(resp["virtual_time_s"])
        if "real_timestamp_ms" in resp:
            real_times.append(float(resp["real_timestamp_ms"]))
        if path=="/measure":
            measure+=1
            ch=int(r["payload"]["channel"])
            if ch!=current_channel:
                switch+=1
            current_channel=ch
            measure_results[resp.get("measure_result")]+=1
        elif path=="/clear":
            clear+=1
            ch=int(r["payload"]["channel"])
            cr=resp.get("clear_result")
            clear_results[cr]+=1
            if cr=="success":
                cleared_channels.add(ch)
            else:
                fail_clear+=1
    duration=(max(real_times)-min(real_times))/1000.0 if real_times else float("nan")
    summary={
        "accepted_actions": len(rows),
        "virtual_time_s": vtime,
        "program_run_duration_s_from_log": duration,
        "move_distance_m": move,
        "measure_calls": measure,
        "clear_calls": clear,
        "channel_switches": switch,
        "clear_failures": fail_clear,
        "cleared_channels": sorted(cleared_channels),
        "cleared_count": len(cleared_channels),
        "measure_results": dict(measure_results),
        "clear_results": dict(clear_results),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
