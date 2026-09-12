# CVR 本地验证运行手册

CVR（Certified Variable Waypoint Routing）允许用公开观测生成变量测点，但每次修改未来扫描计划前，必须逐频道通过独立几何覆盖证书。Q3 使用目标圆盘上的最大最近点距离证书；Q4 使用覆盖目标圆盘的凸包和相关 Delaunay 三角形最大边证书。变量测点永远使用 `coverage_idx=None`，只有坐标严格吻合的原 S3/S25 点才携带历史站点编号。

本轮包含三臂：

- `ISR`：冻结的 ISR-v1；
- `SEGFIXED`：只改变固定任务的分段和顺序，不允许变量点；
- `CVR`：分段调度并允许经过证书的局部变量点替换。

## 环境与版本记录

在仓库根目录运行：

```powershell
python --version
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
git rev-parse HEAD
python -m pytest -q
```

## 开发集

开发集固定使用每题 50 个种子 `160000–160049`，每三场一场 edge，其余 uniform；三臂共 300 次运行。运行器先追加 `started`，随后追加 `complete` 或 `failed`，每局决策日志单独保存，因此可以从完成键安全续跑。

```powershell
python scripts/run_cvr_experiments.py --stage dev --modes 3,4 --arms ISR,SEGFIXED,CVR --out-dir results/cvr/dev_160000_160049
python scripts/audit_cvr_certificates.py results/cvr/dev_160000_160049/decisions
python scripts/analyze_cvr_local.py --runs results/cvr/dev_160000_160049/runs.csv --out-dir results/cvr/dev_160000_160049 --stage dev
```

开发资格必须同时满足全清、费用残差不超过 `1e-6` 秒、平均 `T/N` 至少降低 3%、P95 与最慢局不超过 ISR 的 103%、程序耗时门槛和机制门槛。CVR 必须实际删除物理站点，或同时降低移动距离和总时间；只改变日志或评分不算有效机制。

## 条件独立验收

只有 `summary.json` 对应题号的 `preferred` 非空，才允许提交一个候选到独立验收。命令中的候选必须与开发选择完全一致，运行器会核对代码哈希和种子占用。

```powershell
python scripts/run_cvr_experiments.py --stage acceptance --modes 3 --arms ISR,CVR --development-summary results/cvr/dev_160000_160049/summary.json --out-dir results/cvr/acceptance_q3_20260913
python scripts/run_cvr_experiments.py --stage acceptance --modes 4 --arms ISR,CVR --development-summary results/cvr/dev_160000_160049/summary.json --out-dir results/cvr/acceptance_q4_20260913
```

如果开发选择是 `SEGFIXED`，将相应命令中的 `CVR` 换成 `SEGFIXED`。没有候选通过开发门槛时停止，不追加样本。

## 证据解释

- `runs.csv`：所有开始、完成和失败尝试。统计只读取唯一完成记录，失败记录不会被覆盖。
- `decisions/*.jsonl`：逐局公开状态、测量反馈、计划版本和逐频道证书。
- `summary.json`：配对均值、95%/97.5% bootstrap 区间、尾部和资格门槛。
- `cost_decomposition.json`：移动、测量、切频、失败清除和成功清除的每源秒数。
- `mechanism.json`：候选数、通过证书及实际执行的替换、移除站点、额外测量、重规划和回退。

该手册只授权和描述本地开发及条件独立验收，不创建、启动或恢复任何官方演练批次。有限本地样本全清是实验事实；几何保证依赖题目给定的静止源、固定朝向/接收半径、1000 米最小接收半径和两位小数示向误差外包假设。
