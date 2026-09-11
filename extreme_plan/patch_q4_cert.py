from pathlib import Path
p=Path(r'D:\数学建模\B_RL\brl\g25o.py')
t=p.read_text(encoding='utf-8')
# import certificates
t=t.replace('from .bilateral import K, bearing_clip, clip, solve_bilateral',
            'from .bilateral import K, bearing_clip, clip, solve_bilateral\nfrom .certificates import (cell_index_of_point, grid_cells_intersecting_disk,\n                           q3_absent_cells, q4_absent_cells)',1)
# init arrays in _run_full
old='''    if mode == 3:
        cells, cell_radius = _q3_absence_grid()
    else:
        cells, cell_radius = None, None
'''
new='''    if mode == 3:
        cells, cell_radius = _q3_absence_grid()
        q4_absent = None
    else:
        cells, half = grid_cells_intersecting_disk(200.0, 1800.0)
        cell_radius = half
        q4_absent = {ch: np.zeros(len(cells), dtype=bool) for ch in CHANNELS}
'''
assert old in t; t=t.replace(old,new,1)
# update_absence replace
old='''    def update_absence() -> None:
        if mode != 3 or cells is None:
            return
        for ch in CHANNELS:
            if env.channels[ch].status != "unknown" or ch in proven_absent:
                continue
            if _q3_proven_absent(neg_points[ch], cells, cell_radius):
                env.channels[ch].status = "absent"
                proven_absent.add(ch)
'''
new='''    def update_absence() -> None:
        if cells is None:
            return
        if mode == 3:
            for ch in CHANNELS:
                if env.channels[ch].status != "unknown" or ch in proven_absent:
                    continue
                if _q3_proven_absent(neg_points[ch], cells, cell_radius):
                    env.channels[ch].status = "absent"
                    proven_absent.add(ch)
                    policy.dynamic_absent_channels += 1
            return
        # Q4：位置-方向局部凸包证书；只有全部单元被排除才标记频道不存在
        if q4_absent is None:
            return
        for ch in CHANNELS:
            if env.channels[ch].status != "unknown" or ch in proven_absent:
                continue
            if len(neg_points[ch]) < 3:
                continue
            new = q4_absent_cells(neg_points[ch], cells, cell_radius) & (~q4_absent[ch])
            if new.any():
                q4_absent[ch] |= new
                policy.dynamic_absent_cells += int(new.sum())
            if bool(q4_absent[ch].all()):
                env.channels[ch].status = "absent"
                proven_absent.add(ch)
                policy.dynamic_absent_channels += 1
'''
assert old in t; t=t.replace(old,new,1)
# scan_point filter
old='''    def scan_point(idx: int) -> None:
        p = points[idx]
        chs = [ch for ch in CHANNELS if env.channels[ch].status == "unknown"
               and ch not in proven_absent and idx not in env.channels[ch].scan_points]
        chs.sort(key=lambda c: (c != env.current_channel, c))
        for ch in chs:
            if env.done:
                return
            measure_and_observe(ch, p, coverage_idx=idx, is_refine=False)
        update_absence()
'''
new='''    def scan_point(idx: int) -> None:
        p = points[idx]
        cell_idx = cell_index_of_point(p, cells, cell_radius) if (mode == 4 and cells is not None) else -1
        chs = []
        for ch in CHANNELS:
            if env.channels[ch].status != "unknown" or ch in proven_absent:
                continue
            if idx in env.channels[ch].scan_points:
                continue
            if mode == 4 and q4_absent is not None and cell_idx >= 0 and q4_absent[ch][cell_idx]:
                continue
            chs.append(ch)
        chs.sort(key=lambda c: (c != env.current_channel, c))
        for ch in chs:
            if env.done:
                return
            measure_and_observe(ch, p, coverage_idx=idx, is_refine=False)
        update_absence()
'''
assert old in t; t=t.replace(old,new,1)
# opportunity measure calls in full path update absence after measurement
t=t.replace('''            if d <= 1200.0 and ang > thr:
                measure_and_observe(ch, p, coverage_idx=None, is_refine=True)
                update_absence()''',
'''            if d <= 1200.0 and ang > thr:
                measure_and_observe(ch, p, coverage_idx=None, is_refine=True)
                update_absence()''',1)
p.write_text(t,encoding='utf-8')
print('q4 certificate integrated')