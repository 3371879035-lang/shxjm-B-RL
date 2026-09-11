"""局部环境中的单步 rollout 规划器原型。

每个决策：候选动作在 env 的深拷贝上执行，然后用 G25OR 基策略续行；
若所有候选都不优于直接运行基策略，则回退到基策略。该规划器只在本地
仿真中使用，不接入官方远程 runner。
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from brl.g25o import G25OPolicy
from brl.bilateral import solve_bilateral
from brl.resolver import optical_fallback_points
from brl.local_env import CHANNELS

class RolloutPlanner:
    def __init__(self, coverage='S21', variant='G25OR', max_scan_candidates=6, max_clear_candidates=6):
        self.coverage=coverage; self.variant=variant
        self._base_policy=G25OPolicy(self.variant, coverage=self.coverage)
        self.max_scan_candidates=max_scan_candidates
        self.max_clear_candidates=max_clear_candidates
        self.evaluations=0; self.candidates_tried=0; self.fallbacks=0

    def _candidates(self, env):
        points=G25OPolicy(self.variant,coverage=self.coverage).coverage_set(env.mode)
        scans=[]
        for idx in range(len(points)):
            chs=[ch for ch in CHANNELS if env.channels[ch].status=='unknown' and idx not in env.channels[ch].scan_points]
            if chs:
                travel=float(np.linalg.norm(points[idx]-env.pos))
                scans.append((travel,idx,chs))
        scans.sort(key=lambda t:t[0])
        scans=scans[:self.max_scan_candidates]
        clears=[]
        for ch,st in env.channels.items():
            if st.status!='discovered':
                continue
            if np.isfinite(st.poly_radius) and st.poly_radius<=19.5:
                q=np.asarray(st.poly_center,dtype=float).copy()
                detour=float(np.linalg.norm(q-env.pos))
                clears.append((detour,q,ch))
        clears.sort(key=lambda t:t[0])
        clears=clears[:self.max_clear_candidates]
        return points, scans, clears

    def _run_candidate(self, env, candidate):
        env2=copy.deepcopy(env)
        pts=self._base_policy.coverage_set(env2.mode)
        if hasattr(env2,"coverage_points"):
            env2.coverage_points=pts; env2.n_coverage=len(pts)
        kind=candidate[0]
        if kind=='scan':
            _,idx,chs,p,_=candidate
            for ch in chs:
                if env2.done: break
                env2.measure(p,ch,coverage_idx=idx)
        elif kind=='clear':
            _,q,ch,_,_=candidate
            if not env2.done:
                env2.clear(q,ch)
        pol=G25OPolicy(self.variant,coverage=self.coverage)
        if not env2.done:
            out=pol.run(env2)
        else:
            out={'virtual_time_s':env2.virtual_time}
        return float(out['virtual_time_s'])

    def _resolve_all_discovered(self, env):
        import math
        for ch, st in list(env.channels.items()):
            if st.status != 'discovered' or env.done:
                continue
            first = None; deg = None
            for ob in st.observations:
                if ob.get('result') == 'direction' and ob.get('svd_deg') is not None:
                    first = ob['position']; deg = float(ob['svd_deg']); break
            if first is None:
                continue
            P = st.poly if st.poly is not None and len(st.poly) > 0 else None
            def measure_cb(q, ch=ch):
                out = env.measure(q, ch, is_refine=True)
                return {'measure_result': out.get('measure_result'),
                        'svd_deg': out.get('svd_deg')}
            def clear_cb(q, ch=ch):
                out = env.clear(q, ch)
                return {'clear_result': out.get('clear_result')}
            try:
                solve_bilateral(first, deg, env.pos, measure_cb, clear_cb, initial_region=P)
            except Exception:
                for q in optical_fallback_points(first, deg):
                    if env.channels[ch].status == 'cleared' or env.done:
                        break
                    if env.clear(q, ch).get('clear_result') == 'success':
                        break

    def run(self, env):
        base=G25OPolicy(self.variant,coverage=self.coverage)
        pts=base.coverage_set(env.mode)
        if hasattr(env,"coverage_points"):
            env.coverage_points=pts; env.n_coverage=len(pts)
        while not env.done:
            base_env=copy.deepcopy(env)
            base_out=base.run(base_env) if not base_env.done else {'virtual_time_s':base_env.virtual_time}
            base_total=float(base_out['virtual_time_s'])
            points,scans,clears=self._candidates(env)
            candidates=[]
            for travel,idx,chs in scans:
                candidates.append(('scan',idx,chs,points[idx],travel))
            for detour,q,ch in clears:
                candidates.append(('clear',q,ch,None,detour))
            best=None; best_total=base_total
            for cand in candidates:
                self.evaluations+=1
                try:
                    total=self._run_candidate(env,cand)
                except Exception:
                    continue
                if total < best_total-1e-9:
                    best_total=total; best=cand
            if best is None:
                # 没有候选：先补齐所有已发现源的定位清除，再继续规划
                if any(st.status == 'discovered' for st in env.channels.values()):
                    self.fallbacks += 1
                    self._resolve_all_discovered(env)
                    continue
                # 没有可行动作：回退基策略
                self.fallbacks += 1
                return base.run(env)
            self.candidates_tried+=1
            kind=best[0]
            if kind=='scan':
                _,idx,chs,p,_=best
                for ch in chs:
                    if env.done: break
                    env.measure(p,ch,coverage_idx=idx)
            else:
                _,q,ch,_,_=best
                if not env.done: env.clear(q,ch)
        return {'success':bool(env.success or env.completion_certificate()),'cleared':int(env.cleared_count()),
                'virtual_time_s':float(env.virtual_time),'distance_m':float(env.move_distance),
                'measure_calls':int(env.n_measure),'clear_calls':int(env.n_clear),
                'failed_clear':int(env.n_clear_fail),'solver_failures':0,'optical_fallbacks':0,
                'extra_measurements':0,'planner_candidates':int(self.candidates_tried),
                'planner_evaluations':int(self.evaluations),'planner_fallbacks':int(self.fallbacks)}