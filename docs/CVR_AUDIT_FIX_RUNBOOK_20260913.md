# CVR 修复复跑说明

所有命令在 `D:\数学建模\B_RL` 执行。当前论文、提交包和官方批次不属于本流程。

## 单元与协议检查

```powershell
python -m pytest tests/test_bilateral_state.py tests/test_cvr_policy.py tests/test_cvr_audit.py tests/test_cvr_experiment.py tests/test_cvr_certify.py tests/test_cvr_evidence.py tests/test_cvr_planner.py tests/test_cvr_waypoints.py -q
```

## 40 次策略运行的动作差分

```powershell
python scripts/check_cvr_compatibility.py `
  --out results/cvr/audit_fix_20260913/compatibility_160000_160009.json
```

输出中的 `scenario_pairs=20`、`policy_runs=40`，且 `all_equivalent=true` 才通过。

## 160 条完整后缀分支

```powershell
python scripts/run_cvr_suffix_audit.py `
  --out-dir results/cvr/audit_fix_20260913/suffix_160000_160009_new_attempt
```

脚本不会替换现有 `attempt2` 证据。输出目录必须使用新名称，避免混合两个冻结版本。

## 开发复验

原批次已按用户要求在约 200 局停止，不应恢复成 300 局。若仅验证运行能力，使用单局入口或新的输出目录；不要向已冻结目录追加记录。

完整的原始计划命令如下，仅供复现历史调度，不作为当前建议：

```powershell
python scripts/run_cvr_experiments.py --stage dev --modes 3,4 `
  --arms ISR,COMPAT,CVR `
  --out-dir results/cvr/audit_fix_20260913/dev_recheck_new_attempt
```

审计一个完成批次：

```powershell
python scripts/audit_cvr_runset.py `
  results/cvr/audit_fix_20260913/dev_recheck_160000_160049
```

审计结果分别报告完成日志和未链接/中断日志；后者不能加入性能统计。

