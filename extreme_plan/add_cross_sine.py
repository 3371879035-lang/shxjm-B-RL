from pathlib import Path
p=Path(r'D:\数学建模\B_RL\brl\geometry.py')
t=p.read_text(encoding='utf-8')
if 'def bearing_cross_sine' not in t:
    add='''

def bearing_cross_sine(first: Sequence[float], source_estimate: Sequence[float],
                       candidate: Sequence[float]) -> float:
    """候选源估计位置处两条观测方向的交叉正弦。

    0 表示共线，1 表示正交。旧实现使用候选测点处的夹角，会把共线
    观测误奖励为 180 度；本函数用于替代。
    """
    a = np.asarray(first, dtype=float)
    g = np.asarray(source_estimate, dtype=float)
    p = np.asarray(candidate, dtype=float)
    v1 = a - g
    v2 = p - g
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2))
'''
    t=t.rstrip()+add+'\n'
    p.write_text(t,encoding='utf-8')
    print('added bearing_cross_sine')
else:
    print('already')