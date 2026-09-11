from pathlib import Path
p=Path(r'D:\数学建模\B_RL\brl\coverage.py')
t=p.read_text(encoding='utf-8')
if 'def s21_points' in t:
    print('already')
else:
    add='''

S21_EXACT = np.asarray([
    [0, 0],
    [998, 0], [706, 706], [0, 998], [-706, 706],
    [-998, 0], [-706, -706], [0, -998], [706, -706],
    [1866, 0], [1616, 933], [933, 1616], [0, 1866],
    [-933, 1616], [-1616, 933], [-1866, 0], [-1616, -933],
    [-933, -1616], [0, -1866], [933, -1616], [1616, -933],
], dtype=float)


def s21_points() -> np.ndarray:
    """问题4的21点候选覆盖构造：原点 + 内层8点 + 外层12点。

    坐标必须与整数几何证书保持一致，不要改用三角函数近似环点。
    证书由 scripts/coverage21.py 与 scripts/check_s21_certificate.py 生成和独立校验。
    """
    return S21_EXACT.copy()
'''
    t=t.replace('\nif __name__ == "__main__":', add+'\n\nif __name__ == "__main__":',1)
    # 在 main 中加一行 S21 输出
    t=t.replace('    p4 = s4_points()\n    print("S3", len(p3), p3.tolist())\n    print("S4", len(p4))',
                '    p4 = s4_points()\n    p21 = s21_points()\n    print("S3", len(p3), p3.tolist())\n    print("S4", len(p4))\n    print("S21", len(p21))',1)
    p.write_text(t,encoding='utf-8')
    print('s21 added')