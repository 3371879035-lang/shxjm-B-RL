"""官方问题4 S25/S21 交替演练：每个循环各跑一局，便于控制批次顺序。

每个子进程仍调用 auto_official_g25o_batch.py --runs 1，由它负责点击
开始按钮并等待 runner 完成。所有请求日志和 summary 保存在 out_dir。
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'scripts'/'auto_official_g25o_batch.py'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs',type=int,default=30)
    ap.add_argument('--mode',type=int,default=4,choices=[3,4])
    ap.add_argument('--out-dir',default=str(ROOT/'results'/'extreme_plan'/'official_alt_p4'))
    ap.add_argument('--robot-id',default='202610094088')
    ap.add_argument('--base-url',default='http://127.0.0.1:2026')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    log=(out/'alternating.log').open('a',encoding='utf-8')
    def logmsg(s):
        print(s,flush=True); log.write(s+'\n'); log.flush()
    start=time.time()
    for i in range(1,args.pairs+1):
        for coverage in ('S25','S21'):
            run_dir=out/f'{coverage}_run_{i:03d}'
            run_dir.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,str(BATCH),'--mode',str(args.mode),'--runs','1',
                 '--robot-id',args.robot_id,'--base-url',args.base_url,
                 '--coverage',coverage,'--out-dir',str(run_dir)]
            logmsg(f'[pair {i}] start {coverage}')
            r=subprocess.run(cmd,cwd=str(ROOT),capture_output=True,text=True)
            if r.returncode!=0:
                logmsg(f'[pair {i}] {coverage} FAILED rc={r.returncode} stderr={r.stderr[-500:]}')
                # 关键：不中断，记录失败；正确性由 summary 校验
            else:
                logmsg(f'[pair {i}] {coverage} done')
    logmsg(f'[done] pairs={args.pairs} wall_s={time.time()-start:.1f}')
    log.close()

if __name__=='__main__':
    main()