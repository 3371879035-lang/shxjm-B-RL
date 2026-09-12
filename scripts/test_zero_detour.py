"""零绕行候选点强制单元测试。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from brl.g21a import segment_candidates

def main():
    rng=np.random.default_rng(20260912)
    checked=0
    for _ in range(1000):
        A=rng.uniform(-1500,1500,2); B=rng.uniform(-1500,1500,2)
        centers=[rng.uniform(-1000,1000,2) for _ in range(rng.integers(0,5))]
        if np.linalg.norm(B-A)<1e-6:
            continue
        for q in segment_candidates(A,B,centers):
            lhs=float(np.linalg.norm(A-q)+np.linalg.norm(q-B))
            rhs=float(np.linalg.norm(A-B))
            if lhs-rhs>1e-6:
                raise RuntimeError(f"zero-detour violation {lhs-rhs}")
            checked+=1
    print({"status":"PASS","candidates_checked":checked,"tol_m":1e-6})

if __name__=="__main__": main()