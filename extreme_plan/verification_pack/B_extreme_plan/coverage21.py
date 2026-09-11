"""21 fixed discovery stations and an exact-integer quadtree certificate.
This certifies geometry only. It does NOT establish official simulator speed.
"""
from __future__ import annotations
import argparse, json, time
from collections import Counter
from pathlib import Path

S21 = [(0,0),(998,0),(706,706),(0,998),(-706,706),(-998,0),(-706,-706),(0,-998),(706,-706),
       (1866,0),(1616,933),(933,1616),(0,1866),(-933,1616),(-1616,933),(-1866,0),
       (-1616,-933),(-933,-1616),(0,-1866),(933,-1616),(1616,-933)]

def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
def hull(pts):
    pts=sorted(set(pts))
    if len(pts)<=1:return pts
    lo=[];up=[]
    for p in pts:
        while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=0:lo.pop()
        lo.append(p)
    for p in pts[::-1]:
        while len(up)>=2 and cross(up[-2],up[-1],p)<=0:up.pop()
        up.append(p)
    return lo[:-1]+up[:-1]

def build_certificate(max_depth=14):
    if not 12<=max_depth<=18:raise ValueError('Use max_depth in 12..18')
    scale=2**(max_depth-12)
    pts=[(x*scale,y*scale) for x,y in S21]
    lim2=(999*scale)**2;domain2=(1800*scale)**2
    stack=[(-2048*scale,-2048*scale,2048*scale,2048*scale,0)]
    accepted=[];unresolved=[];counts=Counter();outside=0;max_sq=0
    while stack:
        x0,y0,x1,y1,depth=stack.pop();counts[depth]+=1
        dx=max(x0,0,-x1);dy=max(y0,0,-y1)
        if dx*dx+dy*dy>domain2:outside+=1;continue
        corners=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        nearby=[j for j,(x,y) in enumerate(pts)
                if max((x-qx)**2+(y-qy)**2 for qx,qy in corners)<lim2]
        h=hull([pts[j] for j in nearby])
        if len(h)>=3 and all(cross(h[i],h[(i+1)%len(h)],v)>0 for i in range(len(h)) for v in corners):
            ids=[pts.index(p) for p in h]
            accepted.append([x0,y0,x1,y1,depth,ids])
            max_sq=max(max_sq,max((pts[j][0]-x)**2+(pts[j][1]-y)**2 for j in ids for x,y in corners))
        elif depth>=max_depth:unresolved.append([x0,y0,x1,y1,depth,nearby])
        else:
            mx=(x0+x1)//2;my=(y0+y1)//2
            stack.extend([(x0,y0,mx,my,depth+1),(mx,y0,x1,my,depth+1),
                          (x0,my,mx,y1,depth+1),(mx,my,x1,y1,depth+1)])
    return {'info':{'points':S21,'scale':scale,'max_depth':max_depth,'proof_receive_radius_m':999,
                    'verified_cells':len(accepted),'unresolved_cells':len(unresolved),
                    'outside_cells':outside,'visited_by_depth':dict(counts),
                    'max_support_corner_dist':max_sq**0.5/scale},
            'leaves':accepted,'unresolved':unresolved}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(Path(__file__).parent/'proofs'/'s21_certificate.json'))
    args=ap.parse_args();t=time.perf_counter();cert=build_certificate()
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(cert,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(cert['info'],ensure_ascii=False,indent=2));print('wall_s',time.perf_counter()-t)
    if cert['unresolved']:raise SystemExit('FAILED: unresolved cells remain')
