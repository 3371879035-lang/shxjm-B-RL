from pathlib import Path
base=Path(r'D:\数学建模\数学建模B2-RL\论文')
p=base/'B题_方案B_几何保证主动搜索_论文稿.md'
t=p.read_text(encoding='utf-8')
new_secs=Path(r'D:\数学建模\B_RL\extreme_plan\new_sections.md').read_text(encoding='utf-8')
old='### 10.7 正式测试方案与记录表'
assert old in t
t=t.replace(old,new_secs.rstrip()+'\n',1)
t=t.replace('第10.7节的记录表','第10.11节的记录表')
old='| coverage.py | S3、S4、S25覆盖点构造与核验 |'
assert old in t
new=old+'\n| certificates.py | Q3圆盘并集与Q4局部凸包动态不存在证书 |\n| scripts/audit_q1q2.py | Q1退化分类、可实现反例、MEC最小性与Q2连续最坏距离 |\n| scripts/coverage21.py、check_s21_certificate.py | S21整数四叉树证书与独立检查器 |\n| scripts/check_q3_grid_fix.py、check_q4_certificate.py | Q3边界网格修复与Q4凸包证书回归 |\n| scripts/oracle_bounds.py、evaluate_oracle_gap.py | 开放中心TSP与20米邻域Oracle上下界 |\n| scripts/run_full_ablation.py、planner_rollout.py | 完整滚动版消融与单步rollout规划原型 |'
t=t.replace(old,new,1)
old_rep='''复现顺序：

1. python scripts/verify_math.py
2. python scripts/run_g25o_compare.py --n 100 --start 92000 --variant G25OR
3. python scripts/run_official_g25o.py --mode 3 --robot-id <队号> --variant G25OR
4. python scripts/report_three_g25_variants.py'''
assert old_rep in t
new_rep='''复现顺序：

1. python scripts/verify_math.py
2. python scripts/coverage21.py 与 python scripts/check_s21_certificate.py
3. python scripts/audit_q1q2.py
4. python scripts/check_q3_grid_fix.py 与 python scripts/check_q4_certificate.py
5. python scripts/oracle_bounds.py --self-test
6. python scripts/run_g25o_compare.py --n 100 --start 92000 --variant G25OR --coverage S25
7. python scripts/run_g25o_compare.py --n 100 --start 92000 --variant G25OR --coverage S21
8. python scripts/run_official_g25o.py --mode 3 --robot-id <队号> --variant G25OR
9. python scripts/report_three_g25_variants.py'''
t=t.replace(old_rep,new_rep,1)
old_concl='''5. 官方G25O、G25OR、G25O-R full各100+100局全部清除，三版本平均时间无显著差异；G25OR在保持数学严格外包修复的同时成绩等价，因此作为最终方案。
6. 最终执行采用A0确定性调度，PPO仅作为可选调度消融；PPO在当前实现中没有偏离A0先验，未独立产生速度收益，完成保证由确定性几何层提供。'''
new_concl='''5. 官方G25O、G25OR、G25O-R full各100+100局全部清除，三版本平均时间无显著差异；G25OR在保持数学严格外包修复的同时成绩等价，因此仍作为冻结基线。
6. Q4新增21点覆盖候选S21通过整数几何证书和独立检查，路线仅缩短1.46%，但50对本地严格配对中问题4虚拟时间降低3.45%-3.87%（p<10^-5），全清50/50；官方小样本交替验证全部清除，S21作为正式挑战者保留，最终版本以官方更大样本验证后再定。
7. Q1可实现反例、Q2条件候选区与817.749 m连续最坏距离、Q3边界网格缺口和Q4局部凸包证书均已审计；Oracle上下界给出每源不超过7.75秒的知情差距范围。完整滚动版在修复后仍未优于G25OR，只能说明该实现无额外收益，不能说明调度空间已经饱和。
8. 最终执行采用A0确定性调度，PPO仅作为可选调度消融；PPO在当前实现中没有偏离A0先验，未独立产生速度收益，完成保证由确定性几何层提供。'''
assert old_concl in t
t=t.replace(old_concl,new_concl,1)
p.write_text(t,encoding='utf-8')
print('paper sections updated',len(t))