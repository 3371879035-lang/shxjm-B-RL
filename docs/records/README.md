# 数据记录说明

本目录记录论文和代码的可复现实验数据。所有数值来自实际执行的代码、请求日志或官方模拟器数据库，没有手工编造。

## 1. 官方模拟器演练记录

- 来源：官方模拟器 `practice-statistics-queue.sqlite3`
- 参赛队号/robot_id：202610094088
- 问题3：传统方法14局 + 方案B(RL)20局
- 问题4：传统方法17局 + 方案B(RL)20局
- 所有方案B局均以 `user_exit` 正常结束，完成证书成立后自动调用 `/exit`
- 详细逐局记录：`official_practice_runs.csv`
- 汇总比较：`traditional_vs_planb_summary.csv`
- 原始JSON：`../results/official_practice_40.json`、`../results/traditional_vs_rl_comparison.json`

## 2. 本地严格配对实验

- 文件：`local_paired_A0_vs_planB.csv`
- 问题3/4各100个相同随机场景，A0与方案B使用相同源、频道、半径、类型、方向和固定误差场
- 结果：方案B与A0平均虚拟时间完全相同，因为PPO未偏离A0先验

## 3. 数学校验

- 文件：`math_verification.csv`
- 包含S3/S4覆盖、q±接收上界、122点光学保底、直径交叉验证和最小包围圆算例

## 4. 本地模拟器与mock HTTP联调

本地 `local_env.py` 与 `mock_server.py` 的结果明确标注为非官方数据，只用于算法开发、消融和接口联调。官方正式测试数据尚未执行。正式测试后应新增 `formal_p3_q1.jsonl`、`formal_p3_q2.jsonl`、`formal_p3_q3.jsonl` 等文件，并保持同一统计口径。
