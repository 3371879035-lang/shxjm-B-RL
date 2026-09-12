# JSP 复跑与官方 30 局操作手册

所有命令均在仓库根目录 `D:\数学建模\B_RL` 执行。正式数据使用替代后的训练段 `300000–300255` 和校准段 `301000–301063`；原 `290000` 段因采集器 smoke 占用而废弃。

## 训练数据

Q3 与 Q4 可并行采集。每个目录逐分支写入 `branches.jsonl` 和 `labels.jsonl`，并通过 `progress.json` 断点续跑；相同命令再次执行会核对冻结清单后跳过已有分支。

```powershell
python scripts/collect_jsp_counterfactuals.py --modes 3 --splits train,calibration --out-dir results/jsp/counterfactuals_q3_20260913 --wall-budget-s 14400
python scripts/collect_jsp_counterfactuals.py --modes 4 --splits train,calibration --out-dir results/jsp/counterfactuals_q4_20260913 --wall-budget-s 14400
python scripts/validate_jsp_dataset.py results/jsp/counterfactuals_q3_20260913
python scripts/validate_jsp_dataset.py results/jsp/counterfactuals_q4_20260913
```

## 固定模型与开发集

```powershell
python scripts/train_jsp_ranker.py --data results/jsp/counterfactuals_q3_20260913/labels.jsonl results/jsp/counterfactuals_q4_20260913/labels.jsonl --modes 3,4 --out-dir results/jsp/models_20260913
python scripts/run_jsp_experiments.py --stage dev --modes 3,4 --arms ISR,JSPG,JSPL --model-q3 results/jsp/models_20260913/jsp_q3.joblib --model-q4 results/jsp/models_20260913/jsp_q4.joblib --out-dir results/jsp/dev_final_20260913
python scripts/analyze_jsp_local.py results/jsp/dev_final_20260913
```

开发集报告中的 `development_selection` 是进入独立验收的门。每题最多保留一个候选；没有候选达到相对 ISR-v1 至少 3% 的平均提速时，不运行该题独立验收。

## 独立验收

下面示例以 Q4 的 JSP-L 为冻结候选；若开发集选择 JSP-G，将 `JSPL` 改为 `JSPG` 并删除模型参数。

```powershell
python scripts/run_jsp_experiments.py --stage acceptance --modes 4 --arms ISR,JSPL --model-q4 results/jsp/models_20260913/jsp_q4.joblib --out-dir results/jsp/acceptance_q4_20260913
python scripts/analyze_jsp_local.py results/jsp/acceptance_q4_20260913
```

`summary.json` 的 `acceptance_gates` 必须全部为 `true`。脚本固定运行 200 对空间误差主集、30 对 ±1° 端点压力和 30 对固定 +1° 压力，共 520 次策略运行；不能把多个候选送入这一次验收。

## 官方演练 30 局

只有独立验收通过后才执行。以下命令生成并运行 Q4 的 15 个时间区组，每组包含 ISR-v1 和 JSP 各一局；首个区组直接计入 30 局，不另做 smoke。

JSP-L 首次启动：

```powershell
python scripts/auto_official_isr_experiment.py --out-dir results/jsp/official_q4_jsp30_20260913 --experiment-kind jsp30 --variants ISR,JSP --modes 4 --smoke-runs-per-group 0 --main-runs-per-group 15 --schedule-seed 20260912 --jsp-planner learned --jsp-model results/jsp/models_20260913/jsp_q4.joblib
```

JSP-G 首次启动：

```powershell
python scripts/auto_official_isr_experiment.py --out-dir results/jsp/official_q4_jsp30_20260913 --experiment-kind jsp30 --variants ISR,JSP --modes 4 --smoke-runs-per-group 0 --main-runs-per-group 15 --schedule-seed 20260912 --jsp-planner analytic
```

中断后使用完全相同的命令并在末尾添加 `--resume`。冻结清单、模型哈希、运行顺序或策略代码有变化时，恢复会被拒绝，必须建立新的实验目录。批处理使用跨目录公共锁，不允许同时运行另一个官方 runner；旧 400 局目录缺少当前 `experiment_kind`，不能通过当前恢复检查。

批次完成后运行：

```powershell
python scripts/analyze_official_jsp30.py results/jsp/official_q4_jsp30_20260913
```

官方场景不可复放。15 个区组用于控制批次时段影响，不表示两策略面对相同场景。
