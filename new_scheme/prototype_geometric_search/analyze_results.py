"""Recalculate the paired synthetic validation report; never pool official data.
Use the same per-scenario seed for both strategies. Check run completeness,
unique keys, successes, and the complete virtual-time accounting identity.
"""
from pathlib import Path
import argparse, csv, json, platform
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=str(Path(__file__).with_name('global_ablation_validation.csv')))
    ap.add_argument('--out',default=str(Path(__file__).with_name('paired_validation_summary.json')))
    ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=20260911)
    args=ap.parse_args()
    with open(args.csv,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    keys=[(r['mode'],r['kind'],r['variant'],r['seed']) for r in rows]
    if len(set(keys)) != len(keys): raise ValueError('Duplicate scenario/strategy rows')
    for r in rows:
        t=float(r['distance_m'])/5+5*int(r['measure_calls'])+int(r['switches'])+3*int(r['failed_clear'])+5*int(r['cleared'])
        if not np.isclose(t,float(r['virtual_s']),atol=1e-6,rtol=1e-10):
            raise ValueError(f'Invalid time accounting for {r["seed"]}')
    summaries=[];rng=np.random.default_rng(args.seed)
    for mode in (3,4):
      for kind in ('edge','uniform'):
        ra={int(r['seed']):r for r in rows if int(r['mode'])==mode and r['kind']==kind and r['variant']=='G31'}
        rb={int(r['seed']):r for r in rows if int(r['mode'])==mode and r['kind']==kind and r['variant']=='G25O'}
        if ra.keys()!=rb.keys(): raise ValueError('Unpaired seeds')
        seeds=sorted(ra)
        a=np.array([float(ra[s]['virtual_s']) for s in seeds]);b=np.array([float(rb[s]['virtual_s']) for s in seeds])
        # Keep failures in the report. Do not silently present failed runs
        # as successful completion times.
        fail_a=sum(int(ra[s]['cleared']) != int(ra[s]['sources']) for s in seeds)
        fail_b=sum(int(rb[s]['cleared']) != int(rb[s]['sources']) for s in seeds)
        diff=b-a;n=len(diff)
        boot=diff[rng.integers(0,n,size=(args.bootstrap,n))].mean(axis=1)
        rec=dict(mode=mode,kind=kind,pairs=n,failed_a=fail_a,failed_b=fail_b,
                 mean_a=float(a.mean()),mean_b=float(b.mean()),reduction_pct=float(100*(1-b.mean()/a.mean())),
                 mean_diff=float(diff.mean()),bootstrap_ci_mean_diff=np.quantile(boot,[.025,.975]).tolist(),
                 wins=int((diff < -1e-9).sum()),ties=int((abs(diff)<=1e-9).sum()),losses=int((diff>1e-9).sum()),
                 p95_a=float(np.quantile(a,.95)),p95_b=float(np.quantile(b,.95)))
        summaries.append(rec)
    Path(args.out).write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    meta=dict(scope='synthetic paired, NOT official; G31 and G25O share new bilateral solver',
              bootstrap_seed=args.seed,bootstrap_replicates=args.bootstrap,
              row_count=len(rows),scenario_configurations=len(set((r['mode'],r['kind'],r['seed']) for r in rows)),
              all_cleared_runs=sum(int(r['sources'])==int(r['cleared']) for r in rows),
              python=platform.python_version(),numpy=np.__version__)
    Path(args.out).with_name('analysis_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'metadata':meta,'paired_results':summaries},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
