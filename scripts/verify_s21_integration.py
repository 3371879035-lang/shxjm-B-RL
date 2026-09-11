"""S21 接入一致性检查：坐标、路线长度和随机几何抽样。"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from brl.coverage import s21_points, s25_points
from brl.g25o import route_open
from scripts.coverage21 import S21

def route_length(points):
    order=route_open(np.asarray(points,float),[0,0]); pts=np.vstack([[0,0],np.asarray(points,float)[order]])
    return float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())

def main():
    p21=s21_points(); ref=np.asarray(S21,float)
    coord_ok=bool(p21.shape==(21,2) and np.array_equal(p21,ref))
    rng=np.random.default_rng(20260912); n=300000
    r=1800*np.sqrt(rng.random(n)); th=rng.uniform(0,2*np.pi,n)
    pos=np.column_stack([r*np.cos(th),r*np.sin(th)])
    dirs=rng.normal(size=(n,2)); dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)+1e-15
    rel=p21[None,:,:]-pos[:,None,:]; dist=np.linalg.norm(rel,axis=2)
    visible=(rel@dirs[:,None,:].transpose(0,2,1))[:,:,0]>=-1e-9
    ok=np.any(visible & (dist<=1000.0+1e-9),axis=1)
    miss=int(np.count_nonzero(~ok))
    out={'coordinate_match':coord_ok,'points':int(len(p21)),
         'route_s25_m':route_length(s25_points()),'route_s21_m':route_length(p21),
         'route_shortening_m':route_length(s25_points())-route_length(p21),
         'random_direction_misses':miss,'random_samples':int(n),
         'note':'随机抽样只作交叉检查，连续覆盖保证来自整数四叉树证书。'}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    p=ROOT/'results'/'extreme_plan'/'s21_integration_check.json'; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    if not coord_ok or miss: raise SystemExit(1)
if __name__=='__main__': main()