"""Q3 动态无源网格边界修复检查。"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from brl.certificates import grid_cells_intersecting_disk
from brl.g25o import _q3_absence_grid, _q3_proven_absent

def old_grid(spacing=200.0, limit=1800.0):
    k=int(limit/spacing)+2
    return np.asarray([(i*spacing,j*spacing) for i in range(-k,k+1) for j in range(-k,k+1)
                       if (i*spacing)**2+(j*spacing)**2<=limit*limit+1e-9],float)

def main():
    new_cells, half=grid_cells_intersecting_disk(200.0,1800.0)
    old=old_grid(); r_old=200/math.sqrt(2)
    rng=np.random.default_rng(7); n=200000
    rr=1800*np.sqrt(rng.random(n)); th=rng.uniform(0,2*np.pi,n)
    pts=np.column_stack([rr*np.cos(th),rr*np.sin(th)])
    d_old=np.min(np.linalg.norm(old[None,:,:]-pts[:,None,:],axis=2),axis=1)
    d_new=np.min(np.linalg.norm(new_cells[None,:,:]-pts[:,None,:],axis=2),axis=1)
    witness=np.array([1780.,260.]); w_old=float(np.linalg.norm(old-witness,axis=1).min())
    w_new=float(np.linalg.norm(new_cells-witness,axis=1).min())
    out={'old_cells':int(len(old)),'new_cells':int(len(new_cells)),
         'old_max_nearest_center_distance_m':float(d_old.max()),
         'new_max_nearest_center_distance_m':float(d_new.max()),
         'theoretical_cover_radius_m':half*math.sqrt(2),
         'old_witness_distance_m':w_old,'new_witness_distance_m':w_new,
         'old_gap_exists':bool(w_old>r_old+1e-9),
         'new_full_cover':bool(d_new.max()<=half*math.sqrt(2)+1e-6),
         'synthetic_all_cells_covered':bool(_q3_proven_absent(new_cells,new_cells)),
         'synthetic_no_negatives':bool(_q3_proven_absent(np.empty((0,2)),new_cells)),
         'note':'旧网格漏掉圆边界细片；新网格保留所有与目标圆相交的闭单元。'}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    p=ROOT/'results'/'extreme_plan'/'q3_grid_fix.json'; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    if not(out['old_gap_exists'] and out['new_full_cover'] and out['synthetic_all_cells_covered']): raise SystemExit(1)
if __name__=='__main__': main()