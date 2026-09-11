"""Q4 局部凸包不存在证书单元测试。"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from brl.certificates import grid_cells_intersecting_disk, q3_absent_cells, q4_absent_cells

def main():
    cells,half=grid_cells_intersecting_disk(200.0,1800.0)
    # 目标单元：原点邻域
    idx=int(np.argmin(np.linalg.norm(cells,axis=1)))
    # 三个负测点围绕该单元，且到四角最大距离远小于1000
    neg=np.array([[400.,0.],[-200.,346.4101615137755],[-200.,-346.4101615137755]])
    q4=q4_absent_cells(neg,cells,half)
    q3=q3_absent_cells(neg,cells,half)
    # 远处孤点不能证明原点单元不存在
    far=np.array([[5000.,0.],[5100.,100.],[4900.,-100.]])
    q4_far=q4_absent_cells(far,cells,half)
    out={'target_cell_index':idx,'target_cell_center':cells[idx].tolist(),
         'q4_marks_target':bool(q4[idx]),'q3_marks_target':bool(q3[idx]),
         'q4_far_marks_target':bool(q4_far[idx]),
         'q4_total_cells':int(q4.sum()),'q3_total_cells':int(q3.sum()),
         'note':'Q4证书要求负测点凸包严格包含单元四角且全部在1000m内；远处点不能排除目标单元。'}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    p=ROOT/'results'/'extreme_plan'/'q4_certificate_test.json'; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    if not(q4[idx] and q3[idx] and not q4_far[idx]): raise SystemExit(1)
if __name__=='__main__': main()