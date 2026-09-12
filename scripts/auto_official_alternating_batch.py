"""官方问题4 S25/S21 单进程交替批量演练。

复用 auto_official_g25o_batch 的 UIA 函数，每局切换 coverage，
避免多次启动子进程导致的界面状态问题。
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.auto_official_g25o_batch import (get_sim_window, find_button, click_button,
                                              cleanup_result_dialog)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs',type=int,required=True)
    ap.add_argument('--mode',type=int,default=4,choices=[3,4])
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--robot-id',default='202610094088')
    ap.add_argument('--base-url',default='http://127.0.0.1:2026')
    ap.add_argument('--wait-interface-s',type=float,default=240.0)
    ap.add_argument('--variant',default='G25O')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    log_path=out/'alternating_batch.log'
    log=log_path.open('a',encoding='utf-8')
    def logmsg(s):
        print(s,flush=True); log.write(s+'\n'); log.flush()
    results=[]
    t0=time.time()
    for i in range(1,args.pairs+1):
        for coverage in ('S25','S21'):
            run_dir=out/f'{coverage}_run_{i:04d}'
            run_dir.mkdir(parents=True,exist_ok=True)
            # pre-clean 上一局结果弹窗
            try:
                w0,_=get_sim_window()
                if not find_button(w0,f'开始问题{args.mode}演练测试',timeout=0.5):
                    ok_clean=cleanup_result_dialog(w0,args.mode,timeout=30)
                    if not ok_clean:
                        logmsg(f'[pair {i}] {coverage}: pre-clean failed'); time.sleep(1.0)
            except Exception as e:
                logmsg(f'[pair {i}] {coverage}: pre-clean error {e}')
            log_out=(run_dir/'runner.out.log').open('w',encoding='utf-8')
            log_err=(run_dir/'runner.err.log').open('w',encoding='utf-8')
            cmd=[sys.executable,str(ROOT/'scripts'/'run_official_g25o.py'),
                 '--mode',str(args.mode),'--robot-id',args.robot_id,
                 '--base-url',args.base_url,'--wait-interface-s',str(args.wait_interface_s),
                 '--variant',args.variant,'--coverage',coverage,'--out-dir',str(run_dir)]
            proc=subprocess.Popen(cmd,stdout=log_out,stderr=log_err)
            time.sleep(0.8)
            try:
                w,pid=get_sim_window()
                click_button(w,'关闭公告',timeout=0.5)
                start_text=f'开始问题{args.mode}演练测试'
                if not click_button(w,start_text,timeout=30):
                    proc.kill(); logmsg(f'[pair {i}] {coverage}: start button not found')
                    results.append({'pair':i,'coverage':coverage,'summary':{},'error':'start_button_not_found'})
                    log_out.close(); log_err.close(); continue
            except Exception as e:
                proc.kill(); logmsg(f'[pair {i}] {coverage}: UI error {e}')
                results.append({'pair':i,'coverage':coverage,'summary':{},'error':str(e)})
                log_out.close(); log_err.close(); continue
            logmsg(f'[pair {i}] {coverage}: started, waiting runner')
            deadline=time.time()+360
            while proc.poll() is None and time.time()<deadline:
                time.sleep(0.5)
            if proc.poll() is None:
                proc.kill(); logmsg(f'[pair {i}] {coverage}: runner timeout')
            summary={}
            sp=run_dir/'summary.json'
            if sp.exists():
                summary=json.loads(sp.read_text(encoding='utf-8'))
            results.append({'pair':i,'coverage':coverage,'summary':summary})
            logmsg(f"[pair {i}] {coverage}: success={summary.get('success')} cleared={summary.get('cleared')} V={summary.get('virtual_time_s')}")
            try:
                w,_=get_sim_window()
                cleanup_result_dialog(w,args.mode,timeout=30)
            except Exception as e:
                logmsg(f'[pair {i}] {coverage}: cleanup error {e}'); time.sleep(1.0)
            log_out.close(); log_err.close()
    (out/'alternating_batch_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    logmsg(f'[done] pairs={args.pairs} wall_s={time.time()-t0:.1f}')
    log.close()

if __name__=='__main__':
    main()