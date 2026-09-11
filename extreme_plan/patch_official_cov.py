from pathlib import Path
p=Path(r'D:\数学建模\B_RL\scripts\run_official_g25o.py')
t=p.read_text(encoding='utf-8')
t=t.replace('from brl.coverage import s25_points, s3_points','from brl.coverage import s21_points, s25_points, s3_points',1)
old='''    ap.add_argument("--variant", type=str, default="G25O")
    args = ap.parse_args()'''
new='''    ap.add_argument("--variant", type=str, default="G25O")
    ap.add_argument("--coverage", type=str, default="S25", choices=["S25", "S21", "S4"])
    args = ap.parse_args()'''
assert old in t; t=t.replace(old,new,1)
old='''    belief = RemoteBelief(client, mode=args.mode)
    # G25O 使用新的覆盖点集合，完成证书必须和策略实际扫描的点集一致。
    belief.coverage_points = s3_points() if args.mode == 3 else s25_points()
    belief.n_coverage = len(belief.coverage_points)
    policy = G25OPolicy(args.variant)
    print(f"[enter] remaining_real_duration_s={client.remaining_real_duration_s}", flush=True)
    result = policy.run(belief)
    result.update({"mode": args.mode, "variant": args.variant,
                   "robot_id": args.robot_id, "request_log": str(req_log)})'''
new='''    belief = RemoteBelief(client, mode=args.mode)
    if args.mode == 3:
        cov_points = s3_points()
    elif args.coverage == "S21":
        cov_points = s21_points()
    elif args.coverage == "S4":
        cov_points = __import__("brl.coverage", fromlist=["s4_points"]).s4_points()
    else:
        cov_points = s25_points()
    # 完成证书必须和策略实际扫描的点集一致。
    belief.coverage_points = cov_points
    belief.n_coverage = len(cov_points)
    policy = G25OPolicy(args.variant, coverage=args.coverage)
    print(f"[enter] remaining_real_duration_s={client.remaining_real_duration_s}", flush=True)
    result = policy.run(belief)
    result.update({"mode": args.mode, "variant": args.variant, "coverage": args.coverage,
                   "robot_id": args.robot_id, "request_log": str(req_log)})'''
assert old in t; t=t.replace(old,new,1)
p.write_text(t,encoding='utf-8')
# batch patch
p2=Path(r'D:\数学建模\B_RL\scripts\auto_official_g25o_batch.py')
t2=p2.read_text(encoding='utf-8')
t2=t2.replace('    ap.add_argument("--variant", type=str, default="G25O")','    ap.add_argument("--variant", type=str, default="G25O")\n    ap.add_argument("--coverage", type=str, default="S25", choices=["S25", "S21", "S4"])',1)
t2=t2.replace('''               "--variant", args.variant, "--out-dir", str(run_dir)]''',
              '''               "--variant", args.variant, "--coverage", args.coverage, "--out-dir", str(run_dir)]''',1)
p2.write_text(t2,encoding='utf-8')
print('official runner/batch patched')