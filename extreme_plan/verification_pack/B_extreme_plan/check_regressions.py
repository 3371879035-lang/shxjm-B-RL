"""Small mathematical regressions. These do not certify the entire repository."""
from __future__ import annotations
import json, math
from itertools import combinations
from pathlib import Path
import numpy as np

def run():
    out={}
    # Q1: an actual intersection of three 2-degree wedges equal to an equilateral triangle.
    tri=np.array([[0.,0.],[20.,0.],[10.,10.*math.sqrt(3)]])
    dirs=np.array([[1.,0.],[-.5,math.sqrt(3)/2],[-.5,-math.sqrt(3)/2]])
    stations=tri-600*dirs;bearings=[1.,121.,241.];A=[];b=[]
    for p,theta in zip(stations,bearings):
        th=math.radians(theta);u=np.array([math.cos(th),math.sin(th)]);v=np.array([-u[1],u[0]])
        for n in [-u,v-math.tan(math.pi/180)*u,-v-math.tan(math.pi/180)*u]:A.append(n);b.append(n@p)
    A=np.array(A);b=np.array(b);vertices=[]
    for i,j in combinations(range(len(b)),2):
        M=A[[i,j]]
        if abs(np.linalg.det(M))<1e-10:continue
        z=np.linalg.solve(M,b[[i,j]])
        if np.max(A@z-b)<=1e-8 and not any(np.linalg.norm(z-w)<1e-7 for w in vertices):vertices.append(z)
    verts=np.array(vertices)
    if len(verts)!=3 or any(min(np.linalg.norm(v-tri,axis=1))>1e-7 for v in verts):raise RuntimeError('Q1 wedge example failed')
    D=float(np.linalg.norm(verts[:,None]-verts[None,:],axis=2).max())
    out['q1_realizable_counterexample']={'stations':stations.tolist(),'bearings_deg':bearings,
       'vertices':verts.tolist(),'diameter_m':D,'MEC_radius_m':20/math.sqrt(3),'D_over_2_m':D/2}
    # Q2: each candidate is verified analytically; not min over candidates.
    eps=math.radians(1.01);a=750.;h=300.
    worst=math.sqrt(max(a*a+h*h,1500**2+a*a+h*h-3000*(a*math.cos(eps)-h*math.sin(eps))))
    if worst>=1000:raise RuntimeError('Candidate guarantee failed')
    out['q2_individual_candidates']={'candidates_local':[[750,300],[750,-300]],'error_deg':1.01,
        'each_worst_distance_m':worst,'bound_type':'continuous-sector analytic endpoint bound'}
    # Missing boundary cells in a center-inside-disk construction.
    grid=np.array([(x,y) for x in range(-1800,1801,200) for y in range(-1800,1801,200) if x*x+y*y<=1800**2])
    z=np.array([1780.,260.]);d=float(np.linalg.norm(grid-z,axis=1).min());r=200/math.sqrt(2)
    if not np.linalg.norm(z)<1800 or not d>r:raise RuntimeError('Boundary regression changed')
    out['q3_grid_proof_gap']={'point':z.tolist(),'radius_from_origin_m':float(np.linalg.norm(z)),
       'distance_to_nearest_retained_center_m':d,'assumed_cell_radius_m':r,
       'meaning':'A gap in that enclosure argument; not an observed official missed source.'}
    # Wrong vertex in the information-angle score.
    a=np.array([0.,0.]);c=np.array([1000.,0.]);p=np.array([500.,0.])
    old=math.acos(float(np.dot(a-p,c-p)/(np.linalg.norm(a-p)*np.linalg.norm(c-p))))
    u=a-c;v=p-c;good=abs(u[0]*v[1]-u[1]*v[0])/(np.linalg.norm(u)*np.linalg.norm(v))
    out['angle_vertex_regression']={'first':a.tolist(),'source_estimate':c.tolist(),'candidate':p.tolist(),
      'old_angle_deg':math.degrees(old),'correct_bearing_cross_sine':good,
      'meaning':'Collinear bearings must not be rewarded as maximal angular intersection.'}
    return out
if __name__=='__main__':
    out=run();txt=json.dumps(out,ensure_ascii=False,indent=2);print(txt)
    (Path(__file__).parent/'diagnostics'/'regression_results.json').write_text(txt,encoding='utf-8')
