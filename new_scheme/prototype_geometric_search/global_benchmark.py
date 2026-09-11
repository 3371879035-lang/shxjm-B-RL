"""Paired SYNTHETIC ablation. Not the repository's historical A0 or official generator.
G31, G25, G25+opportunistic share the SAME new bilateral resolver and simulator.
Only existing observation callbacks are supplied to the localization policy.
"""
import math, time, json, argparse, csv
from pathlib import Path
import numpy as np
from bilateral_solver import solve_bilateral, bearing_clip, clip, K


def points31():
    h=950.
    return np.array([(h*(i+j/2),h*np.sqrt(3)*j/2) for i in range(-4,5) for j in range(-4,5)
                     if (h*(i+j/2))**2+(h*np.sqrt(3)*j/2)**2<=(1800+h)**2+1e-8])

def points25():
    a=np.arange(12)*np.pi/6
    return np.concatenate([np.zeros((1,2)),970*np.stack([np.cos(a+np.pi/12),np.sin(a+np.pi/12)],axis=1),
                           1880*np.stack([np.cos(a),np.sin(a)],axis=1)])

def points7():
    a=np.arange(6)*np.pi/3
    return np.concatenate([np.zeros((1,2)),1200*np.stack([np.cos(a),np.sin(a)],axis=1)])

def route(points,start):
    ps=np.concatenate([np.asarray(start,float).reshape(1,2),np.asarray(points,float)],axis=0)
    D=np.linalg.norm(ps[:,None,:]-ps[None,:,:],axis=-1)
    rem=set(range(1,len(ps)));seq=[0]
    while rem:
        i=min(rem,key=lambda k:(D[seq[-1],k],k));seq.append(i);rem.remove(i)
    for _ in range(30):
        improved=False
        for i in range(1,len(seq)-1):
            for j in range(i+1,len(seq)):
                old=D[seq[i-1],seq[i]];new=D[seq[i-1],seq[j]]
                if j+1<len(seq):old+=D[seq[j],seq[j+1]];new+=D[seq[i],seq[j+1]]
                if new<old-1e-8:seq[i:j+1]=seq[i:j+1][::-1];improved=True
        if not improved:break
    return [i-1 for i in seq[1:]]


class Sim:
    def __init__(self,seed,mode,kind='uniform'):
        rng=np.random.default_rng(seed);n=int(rng.integers(10,17));self.truth={}
        for ch in rng.choice(np.arange(1,21),n,replace=False):
            r=1800*np.sqrt(rng.random()) if kind!='edge' else rng.uniform(1650,1800)
            a=rng.uniform(0,2*np.pi);g=r*np.array([math.cos(a),math.sin(a)])
            R=rng.uniform(1000,1500) if kind!='edge' else 1000.
            phi=rng.uniform(0,2*np.pi)
            normal=np.array([math.cos(phi),math.sin(phi)]) if mode==4 and (kind=='edge' or rng.random()<.5) else None
            if kind=='edge' and mode==4:normal=np.array([math.cos(a),math.sin(a)])
            self.truth[int(ch)]=(g,R,normal,rng.uniform(0,20))
        self.pos=np.zeros(2);self.channel=1;self.t=0.;self.d=0.;self.nm=0;self.ns=0;self.nf=0;self.nc=0;self.cleared=set()
    def move(self,p):
        p=np.asarray(p,float);d=float(np.linalg.norm(self.pos-p));self.pos=p.copy();self.d+=d;self.t+=d/5
    def measure(self,p,ch):
        self.move(p);self.nm+=1;self.t+=5+int(ch!=self.channel);self.ns+=int(ch!=self.channel);self.channel=ch
        if ch not in self.truth or ch in self.cleared:return {'measure_result':'no_signal'}
        g,R,n,phase=self.truth[ch];delta=g-self.pos;dist=np.linalg.norm(delta)
        if dist>R+1e-9 or (n is not None and n@(self.pos-g)<-1e-9):return {'measure_result':'no_signal'}
        if dist<=5:return {'measure_result':'near'}
        e=.5*math.sin(self.pos[0]/37+phase)+.5*math.cos(self.pos[1]/43+.3*phase)
        ang=math.degrees(math.atan2(delta[1],delta[0]))
        return {'measure_result':'direction','svd_deg':round((ang+e)%360,2)%360}
    def clear(self,p,ch):
        self.move(p);self.nc+=1;ok=ch in self.truth and ch not in self.cleared and np.linalg.norm(self.pos-self.truth[ch][0])<=20+1e-8
        self.t+=5 if ok else 3;self.nf+=int(not ok)
        if ok:self.cleared.add(ch)
        return {'clear_result':'success' if ok else 'no_target_in_range'}


def run(seed,mode,variant,kind='uniform'):
    t0=time.time();env=Sim(seed,mode,kind)
    # Policy variables (do not access env.truth until evaluation).
    unknown=set(range(1,21));tracks={};done=set();extra=0
    points=points7() if mode==3 else points31() if variant=='G31' else points25()
    order=route(points,[0,0]);visited=0
    def observe(ch,p,obs):
        typ=obs['measure_result']
        if typ=='near':
            assert env.clear(p,ch)['clear_result']=='success';done.add(ch);unknown.discard(ch);tracks.pop(ch,None)
        elif typ=='direction':
            deg=obs['svd_deg'];a=math.radians(deg)
            if ch not in tracks:
                B=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
                P=np.array([[0,0],[1500,-1500*K],[1500,1500*K]])@B.T+p
                for n,c in [(np.array([1.,0.]),1800.),(np.array([-1.,0.]),1800.),(np.array([0.,1.]),1800.),(np.array([0.,-1.]),1800.)]:P=clip(P,n,c)
                tracks[ch]={'first':np.asarray(p).copy(),'deg':deg,'P':P,'nobs':1}
            else:
                tracks[ch]['P']=bearing_clip(tracks[ch]['P'],np.asarray(p),a);tracks[ch]['nobs']+=1
            assert len(tracks[ch]['P'])>0
            unknown.discard(ch)
    for ix in order:
        if not unknown or len(tracks)+len(done)==16:break
        p=points[ix];oldtracks=list(tracks)
        for ch in sorted(unknown,key=lambda c:(c!=env.channel,c)):
            observe(ch,p,env.measure(p,ch))
        visited+=1
        if variant=='G25O':
            for ch in oldtracks:
                if ch not in tracks:continue
                st=tracks[ch];P=st['P'];c=(P.min(axis=0)+P.max(axis=0))/2
                if np.linalg.norm(P-c,axis=1).max()<19.5:continue
                # Lightweight geometric ranking only; NOT a visibility certificate.
                delta=c-p;d=np.linalg.norm(delta)
                w=c-st['first'];w0=np.linalg.norm(w)
                sine=abs(delta[0]*w[1]-delta[1]*w[0])/(d*w0+1e-9)
                if d<=1200 and sine>.15 and st['nobs']<4:
                    observe(ch,p,env.measure(p,ch));extra+=1
    # Remaining channels are absent by a full fixed-cover scan or count=16.
    assert visited==len(points) or not unknown or len(tracks)+len(done)==16
    # Open route through observable region centers, re-optimized after each clear.
    while tracks:
        chs=list(tracks);centers=[(tracks[c]['P'].min(axis=0)+tracks[c]['P'].max(axis=0))/2 for c in chs]
        ch=chs[route(centers,env.pos)[0]];st=tracks[ch]
        res=solve_bilateral(st['first'],st['deg'],env.pos,
                           lambda p:env.measure(p,ch),lambda p:env.clear(p,ch),initial_region=st['P'])
        assert res.success;done.add(ch);del tracks[ch]
    n=len(env.truth)
    return dict(seed=seed,mode=mode,variant=variant,kind=kind,sources=n,cleared=len(env.cleared),
                virtual_s=env.t,distance_m=env.d,measure_calls=env.nm,switches=env.ns,
                failed_clear=env.nf,clear_calls=env.nc,visited=visited,opportunistic=extra,wall_s=time.time()-t0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=100);ap.add_argument('--start',type=int,default=81000)
    ap.add_argument('--out',default='global_ablation.csv');args=ap.parse_args();rows=[]
    for kind in ['uniform','edge']:
      for mode in [3,4]:
        for variant in ['G31','G25','G25O']:
          if mode==3 and variant=='G25':continue
          a=[]
          for seed in range(args.start,args.start+args.n):
            r=run(seed,mode,variant,kind);rows.append(r);a.append(r)
          print(json.dumps(dict(mode=mode,variant=variant,kind=kind,n=len(a),all_clear=sum(x['sources']==x['cleared'] for x in a),mean_s=float(np.mean([x['virtual_s'] for x in a])),mean_per_source_s=float(np.mean([x['virtual_s']/x['sources'] for x in a])),mean_distance_m=float(np.mean([x['distance_m'] for x in a])),mean_measures=float(np.mean([x['measure_calls'] for x in a])),mean_failed_clear=float(np.mean([x['failed_clear'] for x in a])))),flush=True)
    with open(args.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
