from pathlib import Path
p=Path(r'D:\数学建模\B_RL\scripts\run_g25o_compare.py')
t=p.read_text(encoding='utf-8')
t=t.replace('from brl.coverage import s25_points, s25_max_triangle_edge','from brl.coverage import s21_points, s25_points, s25_max_triangle_edge',1)
t=t.replace('def run_g25o(mode, seed, sources, variant="G25O"):\n    n = len(sources)\n    env = RadioEnv(mode=mode, n_sources=n, seed=seed, step_limit=20000)\n    env.reset(seed=seed, n_sources=n, sources=sources)\n    policy = G25OPolicy(variant)\n    out = policy.run(env)\n    out.update({"variant": "G25O", "sources": n})',
            'def run_g25o(mode, seed, sources, variant="G25O", coverage="S25"):\n    n = len(sources)\n    env = RadioEnv(mode=mode, n_sources=n, seed=seed, step_limit=20000)\n    env.reset(seed=seed, n_sources=n, sources=sources)\n    policy = G25OPolicy(variant, coverage=coverage)\n    out = policy.run(env)\n    out.update({"variant": f"{variant}-{coverage}", "sources": n})',1)
t=t.replace('    ap.add_argument("--variant", type=str, default="G25O")\n    args = ap.parse_args()',
            '    ap.add_argument("--variant", type=str, default="G25O")\n    ap.add_argument("--coverage", type=str, default="S25", choices=["S25", "S21", "S4"])\n    args = ap.parse_args()',1)
t=t.replace('                b = run_g25o(mode, seed, src, variant=args.variant)','                b = run_g25o(mode, seed, src, variant=args.variant, coverage=args.coverage)',1)
t=t.replace('                    r = {"kind": kind, "mode": mode, "seed": seed, "method": method, "variant": args.variant}',
            '                    r = {"kind": kind, "mode": mode, "seed": seed, "method": method, "variant": args.variant, "coverage": args.coverage}',1)
t=t.replace('    meta = {"n_per_group": args.n, "start_seed": args.start, "wall_s": time.time() - t0,\n            "s25_points": len(s25_points()), "s25_max_triangle_edge_m": s25_max_triangle_edge(),\n            "rows": len(rows)}',
            '    meta = {"n_per_group": args.n, "start_seed": args.start, "wall_s": time.time() - t0,\n            "variant": args.variant, "coverage": args.coverage,\n            "coverage_points": len(s21_points() if args.coverage == "S21" else s25_points()),\n            "s25_max_triangle_edge_m": s25_max_triangle_edge(), "rows": len(rows)}',1)
p.write_text(t,encoding='utf-8')
print('compare patched')