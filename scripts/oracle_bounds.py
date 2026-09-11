"""Offline clairvoyant bounds; never import scene truth into the official runner.
Input JSON: {"positions": [[x,y], ...]} from an OWN local benchmark scene.
The center-tour optimum is an oracle FEASIBLE upper bound, not a lower bound.
"""
from __future__ import annotations
import argparse, json, math, time
from itertools import permutations
from pathlib import Path
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

@njit(cache=True)
def _held_karp(D):
    n = D.shape[0]-1
    dp=np.full((1<<n,n),np.inf)
    parent=np.full((1<<n,n),-1,np.int32)
    for j in range(n): dp[1<<j,j]=D[0,j+1]
    for mask in range(1,1<<n):
        for j in range(n):
            if not mask & (1<<j): continue
            prev=mask ^ (1<<j)
            if prev==0: continue
            best=np.inf; pred=-1
            for k in range(n):
                if prev & (1<<k):
                    val=dp[prev,k]+D[k+1,j+1]
                    if val<best:best=val;pred=k
            dp[mask,j]=best;parent[mask,j]=pred
    mask=(1<<n)-1;j=int(np.argmin(dp[mask]));cost=dp[mask,j]
    order=np.empty(n,np.int64)
    for t in range(n-1,-1,-1):
        order[t]=j
        k=int(parent[mask,j])
        mask ^= (1<<j)
        if k < 0:
            break
        j=k
    return cost,order

def oracle_bounds(positions, start=(0.0,0.0), clear_radius=20.0, speed=5.0):
    pts=np.asarray(positions,dtype=float)
    if pts.ndim!=2 or pts.shape[1]!=2 or not 1<=len(pts)<=16 or not np.isfinite(pts).all():
        raise ValueError('positions must be 1..16 finite 2-D points')
    if clear_radius<0 or speed<=0:raise ValueError('Invalid radius or speed')
    n=len(pts);coords=np.vstack([np.asarray(start,dtype=float),pts])
    D=np.linalg.norm(coords[:,None]-coords[None,:],axis=2)
    Lcenter,order=_held_karp(D)
    radii=np.r_[0.0,np.full(n,clear_radius)]
    Dlow=np.maximum(0.,D-radii[:,None]-radii[None,:])
    Ledge,_=_held_karp(Dlow)
    Lsimple=max(0.,Lcenter-(2*n-1)*clear_radius)
    # A tiny downward cushion guards insignificant floating point roundoff.
    Llower=max(0.,max(Lsimple,Ledge)-1e-7)
    return {'n':n,'center_tour_length_m':float(Lcenter),
            'center_visit_order_zero_based':order.tolist(),
            'oracle_length_lower_bound_m':float(Llower),
            'oracle_time_lower_bound_s':float(Llower/speed+5*n),
            'oracle_feasible_time_upper_bound_s':float(Lcenter/speed+5*n),
            'oracle_lower_s_per_source':float(Llower/(speed*n)+5),
            'oracle_upper_s_per_source':float(Lcenter/(speed*n)+5),
            'oracle_bracket_width_s_per_source':float((Lcenter-Llower)/(speed*n)),
            'interpretation':'Clairvoyant disk-visiting benchmark, NOT an online completion guarantee.'}

def self_test():
    rng=np.random.default_rng(61273)
    for n in range(1,9):
        p=rng.normal(0,100,(n,2));out=oracle_bounds(p)
        brute=min(sum(np.linalg.norm(p[a]-p[b]) for a,b in zip(seq[:-1],seq[1:]))
                  +np.linalg.norm(p[seq[0]]) for seq in permutations(range(n)))
        if abs(out['center_tour_length_m']-brute)>1e-7:raise RuntimeError('DP test failed')
    return {'held_karp_vs_bruteforce':'PASS for n=1..8',
            'single_source_example':oracle_bounds([[100.,0.]]),
            'bound_note':'For n<=16, the theoretical center/disk bracket is at most 7.75 s/source.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scene');ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--output');args=ap.parse_args();t=time.perf_counter()
    if args.self_test:out=self_test()
    elif args.scene:out=oracle_bounds(json.loads(Path(args.scene).read_text(encoding='utf-8'))['positions'])
    else:ap.error('Pass --scene LOCAL_SCENE.json or --self-test')
    out['wall_seconds_including_optional_jit']=time.perf_counter()-t
    txt=json.dumps(out,ensure_ascii=False,indent=2);print(txt)
    if args.output:Path(args.output).write_text(txt,encoding='utf-8')
