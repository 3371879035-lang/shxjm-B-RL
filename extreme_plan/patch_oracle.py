from pathlib import Path
p=Path(r'D:\数学建模\B_RL\scripts\oracle_bounds.py')
t=p.read_text(encoding='utf-8')
t=t.replace('parent=np.full((1<<n,n),-1,np.int16)','parent=np.full((1<<n,n),-1,np.int32)',1)
old='''    mask=(1<<n)-1;j=int(np.argmin(dp[mask]));cost=dp[mask,j]
    order=np.empty(n,np.int64)
    for t in range(n-1,-1,-1):
        order[t]=j
        k=parent[mask,j];mask^=1<<j;j=k
    return cost,order'''
new='''    mask=(1<<n)-1;j=int(np.argmin(dp[mask]));cost=dp[mask,j]
    order=np.empty(n,np.int64)
    for t in range(n-1,-1,-1):
        order[t]=j
        k=int(parent[mask,j])
        mask ^= (1<<j)
        if k < 0:
            break
        j=k
    return cost,order'''
assert old in t; t=t.replace(old,new,1)
p.write_text(t,encoding='utf-8')
print('oracle patched')