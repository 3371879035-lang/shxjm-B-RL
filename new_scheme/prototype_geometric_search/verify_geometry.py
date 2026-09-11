from pathlib import Path
import math,json
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import unary_union
from global_benchmark import points25,points31,points7,route
from bilateral_solver import EPS,K
p=points25();ts=[]
for k in range(12):
 i=1+k;j=1+(k+1)%12;o=13+k;on=13+(k+1)%12
 ts += [(0,i,j),(o,on,i),(i,j,on)]
maxedge=max(np.linalg.norm(p[t[a]]-p[t[b]]) for t in ts for a in range(3) for b in range(a))
union=unary_union([Polygon(p[list(t)]) for t in ts]);outer=Polygon(p[13:])
assert union.symmetric_difference(outer).area < 1e-6
assert maxedge<1000
rng=np.random.default_rng(20260911);miss=0;nearmax=0.;n=200000
for off in range(0,n,5000):
 a=rng.uniform(0,2*np.pi,5000);r=1800*np.sqrt(rng.random(5000));g=np.stack([r*np.cos(a),r*np.sin(a)],axis=1)
 phi=rng.uniform(0,2*np.pi,5000);normals=np.stack([np.cos(phi),np.sin(phi)],axis=1)
 d=p[None,:,:]-g[:,None,:];dist=np.linalg.norm(d,axis=2)
 visibility=(d*normals[:,None,:]).sum(axis=2)>=-1e-9
 ds=np.where(visibility,dist,np.inf).min(axis=1)
 miss+=int((ds>1000).sum());nearmax=max(nearmax,float(ds.max()))
R={}
for name,pts in [('S7',points7()),('S31',points31()),('S25',points25())]:
 path=pts[route(pts,[0,0])];L=np.linalg.norm(np.diff(np.concatenate([[[0,0]],path]),axis=0),axis=1).sum();R[name]={'n_points':len(pts),'open_route_m':float(L)}
result={'triangles':len(ts),'max_triangle_edge_m':float(maxedge),'outer_apothem_m':1880*math.cos(math.pi/12),'tiling_area_difference_m2':float(union.symmetric_difference(outer).area),'random_cases':n,'misses':miss,'largest_observed_nearest_visible_m':nearmax,'routes_nearest_then_2opt':R,'terminal_two_disk_radius_bound_m':math.sqrt(12**2+(1500*K/2)**2),'bilateral_at_m750_side_m':750*K+5,'proof_eps_deg':math.degrees(EPS)}
json.dump(result,open(str(Path(__file__).with_name('geometry_verified.json')),'w'),indent=2);print(json.dumps(result,indent=2))
