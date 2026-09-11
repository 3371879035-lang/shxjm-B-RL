from pathlib import Path
import sys,math,itertools,json,time
import numpy as np
from bilateral_solver import *
t0=time.time();count=0;worst=[0,0,0];fail=[]
# Endpoint and boundary cases: exact minimum/maximum ranges,
# maximal rounded bearing offsets, field of view edges, zero-/extreme-noise.
for r,ang_deg,phi_delta,noise,geometry in itertools.product(
 [5.00001,6,11.9,12,12.1,19.9,20,20.1,23.4375,375,749.999,750,750.001,999.999,1000,1000.001,1499.999,1500],
 [-1.005,-1.,-0.0001,0.,0.0001,1.,1.005],
 [-90,-89.999999,-60,0,60,89.999999,90],['smooth','extreme','zero'],[False,True]):
    a=math.radians(ang_deg);g=np.array([r*math.cos(a),r*math.sin(a)])
    p=a+math.pi+math.radians(phi_delta)
    env=OneSourceSimulation(g,max(1000.,r),[math.cos(p),math.sin(p)],count+91,noise)
    try:
      rr=solve_bilateral([0,0],0,[0,0],env.measure,env.clear,geometry=geometry)
      assert env.finished
      assert rr.extra_measures<=12 and rr.clear_attempts<=2 and rr.rounds<=6
      worst=[max(worst[0],rr.extra_measures),max(worst[1],rr.clear_attempts),max(worst[2],rr.rounds)]
    except Exception as e:
      fail.append({'r':r,'ang':ang_deg,'delta':phi_delta,'noise':noise,'geometry':geometry,'error':str(e)})
    count+=1
result={'cases':count,'failed':len(fail),'max_extra_measures':worst[0],'max_clear_attempts':worst[1],'max_rounds':worst[2],'wall_s':time.time()-t0,'failures':fail[:20]}
print(json.dumps(result,indent=2));json.dump(result,open(str(Path(__file__).with_name('stress_test_results.json')),'w'),indent=2)
