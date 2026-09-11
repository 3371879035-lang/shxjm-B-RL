from pathlib import Path
p=Path(r'D:\数学建模\B_RL\brl\g25o.py')
t=p.read_text(encoding='utf-8')
# 1 import
t=t.replace('from .coverage import s25_points, s3_points, s4_points',
            'from .coverage import s25_points, s21_points, s3_points, s4_points',1)
# 2 init
old='''    def __init__(self, variant: str = "G25O"):
        self.variant = variant
        self.full_rolling = variant in ("G25O-R", "G25ORFull", "G25ORF", "G25OR+")
        self.rolling = variant in ("G25OR", "G25O-R", "G25OR+") or self.full_rolling
        self.fallback_count = 0
        self.solver_failures = 0
        self.extra_measurements = 0
'''
new='''    def __init__(self, variant: str = "G25O", coverage: str = "S25"):
        self.variant = variant
        self.coverage = str(coverage).upper()
        if self.coverage not in ("S25", "S21", "S4"):
            raise ValueError("coverage must be S25, S21 or S4")
        self.full_rolling = variant in ("G25O-R", "G25ORFull", "G25ORF", "G25OR+")
        self.rolling = variant in ("G25OR", "G25O-R", "G25OR+") or self.full_rolling
        self.fallback_count = 0
        self.solver_failures = 0
        self.extra_measurements = 0
        self.dynamic_absent_channels = 0
        self.dynamic_absent_cells = 0

    def coverage_set(self, mode: int) -> np.ndarray:
        if int(mode) == 3:
            return s3_points()
        if self.coverage == "S21":
            return s21_points()
        if self.coverage == "S4":
            return s4_points()
        return s25_points()
'''
assert old in t; t=t.replace(old,new,1)
# 3 run points
old='''        mode = env.mode
        if mode == 3:
            points = s3_points()
        else:
            points = s25_points() if self.variant in ("G25", "G25O", "G25OR", "G25O-R", "G25OR+") else s4_points()
        order = route_open(points, env.pos)'''
new='''        mode = env.mode
        points = self.coverage_set(mode)
        if hasattr(env, "coverage_points"):
            env.coverage_points = points
            env.n_coverage = len(points)
        order = route_open(points, env.pos)'''
assert old in t; t=t.replace(old,new,1)
# 4 _run_full points
old='''    mode = env.mode
    points = s3_points() if mode == 3 else s25_points()
    unvisited = set(range(len(points)))'''
new='''    mode = env.mode
    points = policy.coverage_set(mode)
    if hasattr(env, "coverage_points"):
        env.coverage_points = points
        env.n_coverage = len(points)
    unvisited = set(range(len(points)))'''
assert old in t; t=t.replace(old,new,1)
# 5 replace q3 grid/functions
old='''def _q3_absence_grid(spacing: float = 200.0):
    global _Q3_CELLS, _Q3_CELL_R
    if _Q3_CELLS is None:
        cells = []
        limit = 1800.0
        k = int(limit / spacing) + 2
        for i in range(-k, k + 1):
            x = i * spacing
            for j in range(-k, k + 1):
                y = j * spacing
                if x * x + y * y <= limit * limit + 1e-9:
                    cells.append((x, y))
        _Q3_CELLS = np.asarray(cells, dtype=float)
        _Q3_CELL_R = spacing * math.sqrt(2.0) / 2.0
    return _Q3_CELLS, _Q3_CELL_R


def _q3_proven_absent(neg_points, cells, cell_radius, margin: float = 5.0) -> bool:
    pts = np.asarray(neg_points, dtype=float)
    if len(pts) == 0:
        return False
    safe_r = 1000.0 - cell_radius - margin
    if safe_r <= 0:
        return False
    for c in cells:
        if not np.any(np.linalg.norm(pts - c, axis=1) <= safe_r):
            return False
    return True
'''
new='''def _circle_rect_min_dist(center, half, radius=1800.0):
    """圆心到轴对齐矩形的最近距离；用于保留与目标圆相交的闭单元。"""
    c = np.asarray(center, dtype=float)
    dx = max(abs(c[0]) - half, 0.0)
    dy = max(abs(c[1]) - half, 0.0)
    return math.hypot(dx, dy)


def _q3_absence_grid(spacing: float = 200.0):
    """返回与目标圆相交的闭正方形单元中心和半边长。

    旧实现只保留中心位于圆内的格子，会在圆边界留下未覆盖细片。
    新实现用圆到矩形最近距离 <= 1800 保留所有相交单元；证书检查时
    使用单元四个角点，而不是中心加外接圆半径。
    """
    global _Q3_CELLS, _Q3_CELL_R
    if _Q3_CELLS is None:
        cells = []
        limit = 1800.0
        half = spacing / 2.0
        k = int(limit / spacing) + 2
        for i in range(-k, k + 1):
            x = i * spacing
            for j in range(-k, k + 1):
                y = j * spacing
                if _circle_rect_min_dist((x, y), half, limit) <= limit + 1e-9:
                    cells.append((x, y))
        _Q3_CELLS = np.asarray(cells, dtype=float)
        _Q3_CELL_R = spacing * math.sqrt(2.0) / 2.0
    return _Q3_CELLS, _Q3_CELL_R


def _q3_proven_absent(neg_points, cells, cell_radius=None, margin: float = 1e-6) -> bool:
    """Q3 全向源不存在证书。

    对一个闭单元，只要存在一个 no_signal 测点 p 满足 p 到单元四个角点
    距离都 <= 1000 - margin，则单元内任意全向源都应被 p 接收，与实测
    矛盾，因此整个单元可排除。所有保留单元都被排除后，频道才标记 absent。
    """
    pts = np.asarray(neg_points, dtype=float)
    if len(pts) == 0 or cells is None or len(cells) == 0:
        return False
    spacing = 200.0
    half = spacing / 2.0
    safe_r = 1000.0 - margin
    corners = np.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float)
    for c in np.asarray(cells, dtype=float):
        cc = c[None, :] + corners
        # 距离的 max 是凸函数，矩形上最大值在角点取到
        d = np.linalg.norm(pts[:, None, :] - cc[None, :, :], axis=2)
        if not np.any(np.max(d, axis=1) <= safe_r):
            return False
    return True
'''
assert old in t; t=t.replace(old,new,1)
# 6 fix geom_angle_bonus
old='''    def geom_angle_bonus(idx: int) -> float:
        p = points[idx]
        bonus = 0.0
        for ch, st in tracks.items():
            c, r = _poly_center_radius(st["P"])
            if not np.isfinite(r):
                continue
            a = st["first"]
            v1 = a - p
            v2 = c - p
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1.0 or n2 < 1.0:
                continue
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            cosang = min(1.0, max(-1.0, cosang))
            ang = math.acos(cosang)
            d = float(np.linalg.norm(p - c))
            if mode == 3:
                if d <= 1300.0:
                    bonus = max(bonus, 450.0 * ang)
            else:
                if d <= 1400.0:
                    bonus = max(bonus, 220.0 * ang)
        return bonus
'''
new='''    def geom_angle_bonus(idx: int) -> float:
        """在候选源估计位置处计算两条观测方向的交叉正弦。

        旧实现计算的是候选测点 p 处的夹角，会把共线观测奖励成 180 度。
        新实现使用 |cross(s1-g, p-g)|/(|s1-g||p-g|)，共线时为 0。
        """
        p = points[idx]
        bonus = 0.0
        for ch, st in tracks.items():
            c, r = _poly_center_radius(st["P"])
            if not np.isfinite(r):
                continue
            g = np.asarray(c, dtype=float)
            a = np.asarray(st["first"], dtype=float)
            v1 = a - g
            v2 = p - g
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            sine = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
            sine = min(1.0, max(0.0, float(sine)))
            ang = math.asin(sine)
            d = float(np.linalg.norm(p - c))
            if mode == 3:
                if d <= 1300.0:
                    bonus = max(bonus, 450.0 * ang)
            else:
                if d <= 1400.0:
                    bonus = max(bonus, 220.0 * ang)
        return bonus
'''
assert old in t; t=t.replace(old,new,1)
# 7 add output metrics
old='''        "optical_fallbacks": int(policy.fallback_count),
        "extra_measurements": int(policy.extra_measurements),
    }'''
new='''        "optical_fallbacks": int(policy.fallback_count),
        "extra_measurements": int(policy.extra_measurements),
        "dynamic_absent_channels": int(policy.dynamic_absent_channels),
        "dynamic_absent_cells": int(policy.dynamic_absent_cells),
    }'''
# replace both return blocks? There are two identical-ish blocks; replace all occurrences
t=t.replace(old,new)
p.write_text(t,encoding='utf-8')
print('g25o patched')