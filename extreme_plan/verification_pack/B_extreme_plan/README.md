# B题极限优化：数学验证包（不是新策略官方成绩）

本包配合《B题极限优化计划_Q1Q2审计与S21覆盖.docx》使用。详见PLAN.md。

## 已实现与未实现

已实现：21点Q4发现覆盖的整数几何证书及独立检查器；自建场景的开放TSP/20米圆邻域oracle上下界；Q1可实现三角形反例、Q2连续扇区最坏距离与两个代码审计见证。
未实现：完整的“证书驱动＋情景rollout”在线策略；未运行本包S21的官方演练；未获得新的200/400秒成绩。
不要把本包的几何测试当作整局策略实验。

## 依赖与命令

Python 3.10+。coverage21.py和check_s21_certificate.py只使用标准库。其余脚本需要numpy。numba可选，用于加速oracle；首次执行包含编译时间。

```bash
python coverage21.py
python check_s21_certificate.py
python check_regressions.py
python oracle_bounds.py --self-test
python oracle_bounds.py --scene LOCAL_SCENE.json --output oracle_result.json
```

LOCAL_SCENE.json示例：`{"positions": [[100,0],[300,100],[200,-500]]}`。只能来自自己可公开用于离线诊断的本地场景，不得将真实场景读取接口注入官方runner。

## S21接入

在coverage21.py中复制精确整数S21坐标，不要用三角函数重新生成近似环点。
同步替换策略点集、RemoteBelief点集、n_coverage、scan_points编号及完成证书。其余定位和调度模块冻结，先做“只换覆盖”的消融。
证书仅证明任意180°定向源在1000米最小接收半径下可被至少一个测点发现；还需要实际对相应频道访问这些点（或已有独立完成证书），且需要可靠清除策略。

## 正确性边界

- 3832个连续单元的证书校验使用整数运算；不是随机抽样覆盖测试。
- oracle中心TSP是知情问题上界，而不是在线问题下界本身；脚本另外给出邻域下界。
- regressions报告full分支网格包围证明缺口，不声称复现了实际官方漏源。
- 所有脚本仅输出到本地，不连接官方模拟器，不写入GitHub，不读取账号信息。
- 计划中的策略选择、时间预算和目标均待端到端实测。

## 文件

- PLAN.md：完整实施计划与参考研究
- coverage21.py / check_s21_certificate.py：独立生成、检查S21证明
- proofs/s21_certificate.json：已通过的计算机辅助覆盖证书
- oracle_bounds.py：开放Held–Karp与20米邻域上下界
- check_regressions.py：Q1/Q2与旧full代码问题的复现
- diagnostics/：实际校验结果、路线长度比较、官方7z只读目录清单与哈希

官方源材料：用户上传题面、使用说明与通信协议。代码审计版本：d6c8906a4ddd4ef3344e5b3d1656c034b032adb1。
