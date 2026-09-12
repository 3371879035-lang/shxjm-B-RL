"""Candidate ISR: integration adapter for the existing repository.
No source truth, simulator RNG, or Oracle modules are imported.
NOT yet validated on the real official simulator. See INTEGRATION.md.
"""
from __future__ import annotations
import math
import numpy as np
from brl.g25o import (route_open, initial_track_polygon as initial_poly,
    _clip_target_square as target_clip, _poly_center_radius as center_radius,
    _best_clear_point as best_clear)
from brl.coverage import s3_points, s25_points
from brl.bilateral import bearing_clip, solve_bilateral
from brl.protocol import (ActionIOError, CertificateViolation,
                          GeometryNumericalError)

class CheckedView:
    _allowed = {"mode","pos","current_channel","channels","n_coverage",
        "coverage_points","done","success","move_distance","virtual_time",
        "n_measure","n_switch","n_clear","n_clear_fail",
        "cleared_count","discovered_count","completion_certificate"}
    def __init__(self,env):object.__setattr__(self,"_env",env)
    def __getattr__(self,key):
        if key not in self._allowed:raise AttributeError("Policy cannot read "+key)
        return getattr(self._env,key)
    def __setattr__(self,key,value):
        if key not in self._allowed:raise AttributeError("Policy cannot set "+key)
        setattr(self._env,key,value)
    def measure(self,position,channel,coverage_idx=None,is_refine=False):
        try:
            out=self._env.measure(position,channel,coverage_idx=coverage_idx,is_refine=is_refine)
            if out.get("accepted") is not True or out.get("measure_result") not in ("direction","near","no_signal"):
                raise ValueError("Invalid measurement response")
            if out["measure_result"]=="direction" and not np.isfinite(float(out.get("svd_deg",float("nan")))):
                raise ValueError("Missing/invalid bearing")
            return out
        except ActionIOError:raise
        except Exception as exc:raise ActionIOError("measure failed; do not issue a different geometric fallback") from exc
    def clear(self,position,channel):
        try:
            out=self._env.clear(position,channel)
            if out.get("accepted") is not True or out.get("clear_result") not in ("success","no_target_in_range"):
                raise ValueError("Invalid clear response")
            return out
        except ActionIOError:raise
        except Exception as exc:raise ActionIOError("clear failed; preserve the original request_id for client retry") from exc

class Baseline:
    def __init__(self,env,*,suffix_tsp=False,global_route=False,try_clear=False):
        self.env=env;self.tracks={};self.points=s3_points() if env.mode==3 else s25_points();env.n_coverage=len(self.points)
        self.order=route_open(self.points,env.pos)
        # Explicitly freeze the tour that reproduces the repository's archived
        # 160000..160049 baseline; raw symmetric floating-point ties vary by platform.
        if env.mode==4:self.order=[0,1,12,11,10,9,8,7,6,5,4,3,2,15,14,13,24,23,22,21,20,19,18,17,16]
        self.suffix_tsp=suffix_tsp;self.global_route=global_route;self.try_clear=try_clear;self.solver_failures=0
    def observe(self,ch,p,obs):
        typ=obs.get('measure_result');env=self.env
        if typ=='near':
            if env.clear(p,ch)['clear_result']=='success':self.tracks.pop(ch,None)
            return
        if typ=='direction' and obs.get('svd_deg') is not None:
            if ch not in self.tracks:self.tracks[ch]=dict(first=p.copy(),deg=float(obs['svd_deg']),P=initial_poly(p,float(obs['svd_deg'])),nobs=1)
            else:
                st=self.tracks[ch];st['P']=target_clip(bearing_clip(st['P'],p,math.radians(float(obs['svd_deg']))));st['nobs']+=1
            if not len(self.tracks[ch]['P']):self.tracks[ch]['P']=initial_poly(self.tracks[ch]['first'],self.tracks[ch]['deg'])
    def scan(self,idx):
        env=self.env;p=self.points[idx];old=list(self.tracks)
        chs=[c for c,s in env.channels.items() if s.status=='unknown' and idx not in s.scan_points];chs.sort(key=lambda c:(c!=env.current_channel,c))
        for ch in chs:
            if env.done:return
            self.observe(ch,p,env.measure(p,ch,coverage_idx=idx))
        for ch in old:
            if ch not in self.tracks or env.done:continue
            st=self.tracks[ch];c,r=center_radius(st['P'])
            if r<=19.5 or st['nobs']>=4:continue
            delta=c-p;d=float(np.linalg.norm(delta));w=c-st['first'];w0=float(np.linalg.norm(w))
            if w0<1e-9:continue
            sine=abs(delta[0]*w[1]-delta[1]*w[0])/(d*w0+1e-9)
            if d<=1200 and sine>.15:self.observe(ch,p,env.measure(p,ch,is_refine=True))
    def resolve(self,ch):
        env=self.env;st=self.tracks[ch]
        if self.try_clear:
            c,r=center_radius(st['P'])
            if 19.5<r<=80:
                if env.clear(c,ch)['clear_result']=='success':self.tracks.pop(ch,None);return
        try:solve_bilateral(st['first'],st['deg'],env.pos,lambda q:env.measure(q,ch,is_refine=True),lambda q:env.clear(q,ch),initial_region=st['P'])
        except ActionIOError:
            # Transport/protocol failures are not geometric evidence.  Propagate
            # them immediately so the caller can retry the same request id and
            # the policy cannot issue a different action after an uncertain one.
            raise
        except GeometryNumericalError:
            self.solver_failures+=1
            # Certified optical strip cover: 2 rows x 61 columns; independent final fallback.
            a=math.radians(st['deg']);u=np.array([math.cos(a),math.sin(a)]);v=np.array([-u[1],u[0]])
            for x in np.linspace(0,1500,61):
                for y in (13.3,-13.3):
                    if env.done:break
                    if env.clear(st['first']+x*u+y*v,ch)['clear_result']=='success':break
                if env.channels[ch].status=='cleared':break
        if env.channels[ch].status!='cleared':
            raise CertificateViolation('geometric fallback failed: not a completion certificate')
        self.tracks.pop(ch,None)
    def run(self):
        env=self.env
        for oi,idx in enumerate(self.order):
            if env.done or env.cleared_count()+env.discovered_count()>=16:break
            self.scan(idx)
            if env.done:break
            if self.tracks and oi+1<len(self.order):
                B=self.points[self.order[oi+1]];best=None
                for ch,st in list(self.tracks.items()):
                    q=best_clear(st['P'],env.pos,B)
                    if q is None:continue
                    det=float(np.linalg.norm(env.pos-q)+np.linalg.norm(q-B)-np.linalg.norm(env.pos-B))
                    if det<=250 and (best is None or det<best[0]):best=(det,ch,q)
                if best is not None:
                    _,ch,q=best
                    if env.clear(q,ch)['clear_result']=='success':self.tracks.pop(ch,None)
        while self.tracks and not env.done:
            chs=list(self.tracks)
            if self.suffix_tsp:
                centers=np.array([center_radius(self.tracks[c]['P'])[0] for c in chs]);ch=chs[route_open(centers,env.pos)[0]]
            else:ch=min(chs,key=lambda c:float(np.linalg.norm(center_radius(self.tracks[c]['P'])[0]-env.pos)))
            self.resolve(ch)
        for _ in range(5):
            if env.done:break
            need=next((idx for idx in self.order if any(s.status=='unknown' and idx not in s.scan_points for s in env.channels.values())),None)
            if need is None:break
            self.scan(need)
            while self.tracks and not env.done:self.resolve(next(iter(self.tracks)))
        return env

class JointRoute(Baseline):
    def __init__(self,env,*,max_region_radius=300.,include_uncertain=True,try_clear=False):
        super().__init__(env,suffix_tsp=True,try_clear=try_clear)
        self.max_region_radius=float(max_region_radius);self.include_uncertain=include_uncertain
        self.service_jobs=0;self.scanning_jobs=0
    def run(self):
        env=self.env;unvisited=set(range(len(self.points)))
        # Zero-distance initial survey. Only real measurements create evidence.
        self.scan(0);unvisited.remove(0)
        steps=0
        while not env.done and steps<100:
            steps+=1
            if env.cleared_count()+env.discovered_count()>=16:unvisited.clear()
            else:
                unvisited={j for j in unvisited if any(s.status=='unknown' and j not in s.scan_points for s in env.channels.values())}
            jobs=[('scan',j) for j in sorted(unvisited)]
            coords=[self.points[j] for _,j in jobs]
            for ch,st in self.tracks.items():
                c,r=center_radius(st['P'])
                if not unvisited or r<=self.max_region_radius:
                    jobs.append(('resolve',ch));coords.append(c)
            if not jobs:
                if self.tracks:
                    chs=list(self.tracks);coords=[center_radius(self.tracks[c]['P'])[0] for c in chs];self.resolve(chs[route_open(coords,env.pos)[0]]);continue
                break
            seq=route_open(np.asarray(coords),env.pos);typ,j=jobs[seq[0]]
            if typ=='scan':self.scan(j);unvisited.remove(j);self.scanning_jobs+=1
            else:self.resolve(j);self.service_jobs+=1
        # Terminal certificate check, never blind exit.
        if not env.done:
            for j in self.order:
                if any(s.status=='unknown' and j not in s.scan_points for s in env.channels.values()):self.scan(j)
            while self.tracks and not env.done:
                chs=list(self.tracks);cs=[center_radius(self.tracks[c]['P'])[0] for c in chs];self.resolve(chs[route_open(cs,env.pos)[0]])
        return env

class JointShell(JointRoute):
    """Observation-triggered outward-first joint service; retains every obligation."""
    def __init__(self,env,threshold=0,probe=True,max_fast_steps=100):
        super().__init__(env,max_region_radius=300.,try_clear=probe);self.threshold=threshold
        self.max_fast_steps=int(max_fast_steps)
    def run(self):
        env=self.env;unvisited=set(range(len(self.points)));self.scan(0);unvisited.remove(0)
        outer_first=env.mode==4 and env.discovered_count()<=self.threshold
        steps=0
        while not env.done and steps<self.max_fast_steps:
            steps+=1
            if env.cleared_count()+env.discovered_count()>=16:unvisited.clear()
            else:unvisited={j for j in unvisited if any(s.status=='unknown' and j not in s.scan_points for s in env.channels.values())}
            outer={j for j in unvisited if j>=13}
            active=outer if outer_first and outer else unvisited
            jobs=[('scan',j) for j in sorted(active)];coords=[self.points[j] for _,j in jobs]
            for ch,st in self.tracks.items():
                c,r=center_radius(st['P'])
                if not unvisited or r<=self.max_region_radius:jobs.append(('resolve',ch));coords.append(c)
            if not jobs:
                if self.tracks:
                    chs=list(self.tracks);coords=[center_radius(self.tracks[c]['P'])[0] for c in chs];self.resolve(chs[route_open(coords,env.pos)[0]]);continue
                break
            typ,j=jobs[route_open(np.asarray(coords),env.pos)[0]]
            if typ=='scan':self.scan(j);unvisited.remove(j)
            else:self.resolve(j)
        if not env.done:
            for j in self.order:
                if any(s.status=='unknown' and j not in s.scan_points for s in env.channels.values()):self.scan(j)
            while self.tracks and not env.done:
                chs=list(self.tracks);cs=[center_radius(self.tracks[c]['P'])[0] for c in chs];self.resolve(chs[route_open(cs,env.pos)[0]])
        return env

class IndependentCandidate:
    def __init__(self,mode,*,probe=True,candidate_name=None):
        self.mode=int(mode)
        if self.mode not in (3,4):raise ValueError("mode must be 3 or 4")
        self.probe=bool(probe)
        self.candidate_name=candidate_name or ("ISR-Q3r800-Q4r300-origin0-20260912" if self.probe else "ISR-NoProbe-safe-20260912")
    def run(self,env):
        if int(env.mode)!=self.mode:raise ValueError("mode mismatch")
        view=CheckedView(env)
        policy=JointShell(view,threshold=0,probe=self.probe)
        policy.max_region_radius=800.0 if self.mode==3 else 300.0
        view.coverage_points=policy.points.copy()
        view.n_coverage=len(policy.points)
        policy.run()
        success=bool(view.completion_certificate())
        if not success:raise CertificateViolation("candidate returned without a completion certificate")
        return {"success":success,"cleared":int(view.cleared_count()),
            "virtual_time_s":float(view.virtual_time),"distance_m":float(view.move_distance),
            "measure_calls":int(view.n_measure),"switches":int(view.n_switch),
            "clear_calls":int(view.n_clear),"failed_clear":int(view.n_clear_fail),
            "solver_failures":int(policy.solver_failures),"optical_fallbacks":int(policy.solver_failures),
            "candidate":self.candidate_name}


class IndependentCandidateNoProbe(IndependentCandidate):
    """Counterfactual arm: ISR-v1 with only the early MEC probe disabled."""

    def __init__(self, mode):
        super().__init__(mode, probe=False, candidate_name="ISR-NoProbe-safe-20260912")
