"""B-question prototype: certified bilateral interval localization.
Only measure/clear feedback is used by the solver. No hidden source access.
Coordinates/angles follow the supplied CUMCM2026B protocol.
This is a local prototype, not a verified official entry.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
import numpy as np

EPS_DEG = 1.01  # conservative: 1 degree sensor error + 0.005 degree rounding
EPS = math.radians(EPS_DEG)
K = math.tan(EPS)
R_MAX = 1500.0


def clip(poly: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Closed halfplane normal @ z <= offset, preserving outer feasibility."""
    if len(poly) == 0:
        return poly
    vals = poly @ normal - offset
    out=[]
    for i, a in enumerate(poly):
        b=poly[(i+1)%len(poly)]; da=vals[i]; db=vals[(i+1)%len(poly)]
        if da <= 1e-9: out.append(a)
        if (da < -1e-9 and db > 1e-9) or (da > 1e-9 and db < -1e-9):
            out.append(a + (b-a)*(da/(da-db)))
    return np.asarray(out, dtype=float).reshape(-1,2)


def bearing_clip(poly: np.ndarray, p: np.ndarray, theta: float) -> np.ndarray:
    u=np.array([math.cos(theta),math.sin(theta)])
    v=np.array([-u[1],u[0]])
    for n in (-u, v-K*u, -v-K*u):
        poly=clip(poly,n,float(n@p))
    return poly


@dataclass
class SolveResult:
    success: bool
    extra_measures: int
    clear_attempts: int
    rounds: int
    position: np.ndarray
    region: np.ndarray


def solve_bilateral(first_position, first_bearing_deg: float, current_position,
                    measure: Callable, clear: Callable, *, geometry=True, initial_region=None) -> SolveResult:
    """Resolve ONE confirmed source through callbacks already bound to its channel.

    measure(np.ndarray)->{'measure_result':str, 'svd_deg':float iff direction}
    clear(np.ndarray)->{'clear_result':'success'|'no_target_in_range'}

    Most episodes use fewer than the upper bound of 12 additional measurements.
    At most two terminal optical probes are needed under the documented model.
    Raise on protocol or consistency errors rather than silently declaring success.
    """
    origin=np.asarray(first_position,float)
    a0=math.radians(float(first_bearing_deg)); u=np.array([math.cos(a0),math.sin(a0)])
    v=np.array([-u[1],u[0]]); B=np.stack([u,v],axis=1)
    pos=np.asarray(current_position,float).copy()
    lo,hi=0.,R_MAX
    poly=np.array([[0.,0.],[R_MAX,-R_MAX*K],[R_MAX,R_MAX*K]])
    if initial_region is not None:
        supplied=(np.asarray(initial_region,float)-origin)@B
        for n,c in ((np.array([-1.,0.]),0.),(np.array([1.,0.]),R_MAX),
                    (np.array([-K,1.]),0.),(np.array([-K,-1.]),0.)):
            supplied=clip(supplied,n,c)
        if len(supplied)==0:raise RuntimeError('Invalid initial conservative region')
        poly=supplied
        lo=max(lo,float(poly[:,0].min()));hi=min(hi,float(poly[:,0].max()))
    nm=nc=rounds=0
    def glob(z):return origin+B@np.asarray(z)
    def ret(ok):return SolveResult(ok,nm,nc,rounds,pos.copy(),poly@B.T+origin)
    for _ in range(7):
        if geometry:
            lo=max(lo,float(poly[:,0].min())); hi=min(hi,float(poly[:,0].max()))
            center=(poly.min(axis=0)+poly.max(axis=0))/2
            if float(np.linalg.norm(poly-center,axis=1).max()) <= 19.5:
                pos=glob(center); nc+=1
                ans=clear(pos)
                if ans.get('clear_result')=='success': return ret(True)
                raise RuntimeError('Certified optical clear failed; check model/geometry/protocol')
        if hi-lo <= 24.+1e-7:
            break
        rounds+=1
        mid=(lo+hi)/2
        side=mid*K+5.
        probes=[np.array([mid,side]),np.array([mid,-side])]
        probes.sort(key=lambda q:float(np.linalg.norm(glob(q)-pos)))
        observed=False
        for q in probes:
            pos=glob(q); nm+=1
            ans=measure(pos)
            status=ans.get('measure_result')
            if status=='near':
                nc+=1
                if clear(pos).get('clear_result')=='success':return ret(True)
                raise RuntimeError('near clear failed')
            if status=='no_signal': continue
            if status!='direction' or 'svd_deg' not in ans:
                raise RuntimeError(f'Invalid measure response: {ans}')
            observed=True
            theta=math.radians(float(ans['svd_deg'])-first_bearing_deg)
            cx=math.cos(theta)
            if cx>math.sin(EPS)+1e-12:lo=max(lo,mid)
            elif cx<-math.sin(EPS)-1e-12:hi=min(hi,mid)
            else:
                w=(R_MAX*K+side)*math.tan(2*EPS)
                lo=max(lo,mid-w);hi=min(hi,mid+w)
            if geometry:poly=bearing_clip(poly,q,theta)
            break
        if not observed:
            # BOTH negative: if x>=mid, a convex combination of the two
            # probes lies on the visible ray, and BOTH probes are closer
            # than the first positive position. Contradiction.
            hi=min(hi,mid)
        poly=clip(poly,np.array([1.,0.]),hi)
        poly=clip(poly,np.array([-1.,0.]),-lo)
        if len(poly)==0 or lo>hi+1e-7:
            raise RuntimeError('Empty certified region: model or numerical inconsistency')
    if hi-lo>24.+1e-6:
        raise RuntimeError('Interval did not contract within six rounds')
    # Cover [lo,hi] x [-hi*K,+hi*K] by TWO 20m optical disks.
    targets=[np.array([(lo+hi)/2,hi*K/2]),np.array([(lo+hi)/2,-hi*K/2])]
    targets.sort(key=lambda q:float(np.linalg.norm(glob(q)-pos)))
    for q in targets:
        pos=glob(q);nc+=1
        if clear(pos).get('clear_result')=='success':return ret(True)
    raise RuntimeError('Two-disk terminal cover failed')


class OneSourceSimulation:
    """Synthetic local environment; deliberately NOT the official generator."""
    def __init__(self, g, radius, normal, seed=0, noise='smooth'):
        self.g=np.asarray(g,float);self.radius=float(radius)
        self.normal=None if normal is None else np.asarray(normal,float)
        self.seed=seed;self.noise=noise
        self.pos=np.zeros(2);self.distance=0.;self.t=0.;self.nm=0;self.nf=0;self.nc=0
        self.finished=False
    def move(self,p):
        p=np.asarray(p,float);d=float(np.linalg.norm(p-self.pos));self.distance+=d;self.t+=d/5;self.pos=p.copy()
    def measure(self,p):
        self.move(p);self.nm+=1;self.t+=5
        d=float(np.linalg.norm(self.g-self.pos))
        visible=(d<=self.radius+1e-9 and (self.normal is None or self.normal@(self.pos-self.g)>=-1e-9))
        if not visible:return {'measure_result':'no_signal'}
        if d<=5:return {'measure_result':'near'}
        t=math.degrees(math.atan2(*(self.g-self.pos)[::-1]))
        x,y=self.pos
        if self.noise=='smooth':e=.5*math.sin(x/37+self.seed)+.5*math.cos(y/43+.31*self.seed)
        elif self.noise=='extreme':e=1. if math.sin(x*1.71+y*2.13+self.seed)>=0 else -1.
        else:e=0.
        return {'measure_result':'direction','svd_deg':round((t+e)%360,2)%360}
    def clear(self,p):
        self.move(p);self.nc+=1
        ok=float(np.linalg.norm(self.pos-self.g))<=20+1e-9 and not self.finished
        self.t+=5 if ok else 3
        self.nf+=int(not ok);self.finished|=ok
        return {'clear_result':'success' if ok else 'no_target_in_range'}


def sample_case(seed, directional=True, boundary=False, noise='smooth'):
    rng=np.random.default_rng(seed)
    radius=float(rng.uniform(1000,1500));r=float(rng.uniform(5.01,radius))
    ang=float(rng.uniform(-math.radians(1),math.radians(1)))
    g=r*np.array([math.cos(ang),math.sin(ang)])
    phi=ang+math.pi+(math.pi/2*(1 if seed%2 else -1) if boundary else rng.uniform(-math.pi/2,math.pi/2))
    normal=np.array([math.cos(phi),math.sin(phi)]) if directional else None
    return OneSourceSimulation(g,radius,normal,seed,noise)


if __name__=='__main__':
    import argparse,json,time
    ap=argparse.ArgumentParser();ap.add_argument('--cases',type=int,default=2000)
    ap.add_argument('--out',default='bilateral_test.json');args=ap.parse_args()
    t0=__import__('time').time();allrows=[]
    for name,directional,boundary,noise in [('omni',False,False,'smooth'),('directional',True,False,'smooth'),('directional_boundary',True,True,'extreme')]:
        rows=[]
        for seed in range(args.cases):
            env=sample_case(seed+20260911,directional,boundary,noise)
            res=solve_bilateral([0,0],0,[0,0],env.measure,env.clear)
            rows.append([env.finished,env.t,env.distance,env.nm,env.nf,res.rounds,env.nc])
        a=np.array(rows)
        r={'scenario':name,'cases':len(rows),'all_cleared':int(a[:,0].sum()),'mean_s':float(a[:,1].mean()),'p95_s':float(np.quantile(a[:,1],.95)),'max_s':float(a[:,1].max()),'mean_distance_m':float(a[:,2].mean()),'mean_extra_measures':float(a[:,3].mean()),'max_extra_measures':int(a[:,3].max()),'max_clear_attempts':int(a[:,6].max()),'mean_failed_clear':float(a[:,4].mean())}
        allrows.append(r);print(json.dumps(r))
    json.dump({'scope':'single-source local; first positive at origin GIVEN, not counted; no channel switching','epsilon_deg':EPS_DEG,'results':allrows,'wall_s':__import__('time').time()-t0},open(args.out,'w'),indent=2)
