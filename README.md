# B题：无线电干扰源快速自动定位与清除（方案B：强化学习调度）

本目录给出 2026 高教社杯 B 题的完整实现：数学建模、几何保证模块、覆盖发现证明、有限步骤光学保底、本地随机环境、带动作屏蔽的 PPO，以及官方模拟器 HTTP+JSON 适配器。

方案B不是用 RL 重新学习测向几何，而是让 PPO 在共享的确定性保证层之上学习调度：下一处扫描点、先处理哪个源、继续交会还是转清除/光学保底。几何、频道账本、覆盖证书、动作屏蔽和有限步骤保底由共同模块给出。

## 1. 目录结构
```
brl/
  geometry.py        # 半平面交、直径、凸多边形、最小包围圆
  coverage.py        # 问题3的7点覆盖 S3、问题4的31点覆盖 S4
  resolver.py        # 第2问第二测点、局部交会候选、122点光学保底
  local_env.py       # 本地训练/评估环境（固定误差场、全向/定向源、宏动作）
  policy.py          # 带动作屏蔽的 actor-critic 网络（96槽位）
  ppo.py             # PPO、行为克隆、GAE、A0先验正则
  evaluate.py        # 配对评估、统计指标
  remote.py          # 官方接口客户端、远程账本、远程宏动作包装器
scripts/
  verify_math.py     # 数学校验
  train_ppo.py       # 训练入口
  evaluate_pair.py   # 本地A0 vs PPO配对评估
  mock_server.py     # 本地mock模拟器，用于联调官方适配器
  run_official.py    # 正式/演练测试入口
results/             # 训练日志、评估结果、模型、图表
problem_spec.md      # 正式题面/附件/协议整理
paper/               # 论文稿与生成脚本
config/team.json     # 参赛队号 202610094088
```
## 2. 安装与运行

依赖：Python 3.10+，numpy scipy torch matplotlib pandas python-docx。

```powershell
cd D:\数学建模\B_RL

# 数学校验
python scripts\verify_math.py

# 训练问题3/4的PPO
python scripts\train_ppo.py --mode 3 --steps 1200 --rollout 256 --bc-episodes 20 --bc-epochs 3 --bc-coef 1.0 --prior-logit 5.0 --lr 1e-4 --save results\ppo_mode3_safe.pt
python scripts\train_ppo.py --mode 4 --steps 1000 --rollout 256 --bc-episodes 20 --bc-epochs 3 --bc-coef 1.0 --prior-logit 5.0 --lr 1e-4 --save results\ppo_mode4_safe.pt

# 本地配对评估
python scripts\evaluate_pair.py --mode 3 --model results\ppo_mode3_safe.pt --seeds 6000-6099
python scripts\evaluate_pair.py --mode 4 --model results\ppo_mode4_safe.pt --seeds 6000-6099
```
## 3. 官方模拟器运行

本队参赛队号为 202610094088，已写入 config/team.json；运行 run_official.py 时若不传 --robot-id，会自动使用该队号。也可直接运行 scripts/run_q3.ps1 或 scripts/run_q4.ps1。

先启动官方模拟器并登录，进入问题3或问题4演练/正式测试，等待5秒倒计时结束、界面提示接口已就绪。然后运行：

```powershell
python scripts\run_official.py --mode 3 --base-url http://127.0.0.1:2026 --strategy hybrid --model results\ppo_mode3_safe.pt --log results\official_q3.jsonl
```

strategy 可选：
- heuristic：仅用 A0 确定性保底策略；
- ppo：直接用 PPO 动作（动作屏蔽和A0先验仍在）；
- hybrid：默认安全门，PPO 偏离 A0 先验且概率优势不足时采用 A0 动作。

## 4. 数学保证

- 问题1：示向误差锥的3个半平面与目标圆域外切多边形求交；枚举顶点对得直径；最小包围圆判断一次清除是否可保证。
- 问题2：全向源的可靠第二测点候选区域 Q_vis = { q : max_{z in F} ||q-z|| <= 1000 }；可直接使用的起点 q± = s1 + 750u ± 300v，最坏接收距离约 817.65m。
- 问题3：7点覆盖（原点 + 半径1200正六边形），最坏发现距离约 968.90m。
- 问题4：边长 950m 的等边三角网格共31点，对任意180°定向源至少一点可见。
- 单源保底：一次有效示向后，2×61 个清除点覆盖长度1500m、半宽<26.18m的矩形，最远约18.11m<20m。
## 5. 完成条件

程序只有在以下证书成立时才调用 /exit：

1. 已成功清除至少 16 个不同频道（由题面上限推出任务完成）；或
2. 20 个频道均已处于“已清除”或“已由覆盖证明判定不存在”。

RL 网络不能自行提前退出；所有动作屏蔽与有限额度（每源最多 6 次额外局部测量）都在网络之外实现。

## 6. 专用技能与质量 Gate

本项目同时按 radio-interference-b-modeling 专用技能执行：问题一路由到几何 reference，问题二使用信息收益/代价权衡，问题三覆盖搜索与频道账本，问题四保留 no_signal 的多重解释。旋转卡壳直径已与暴力顶点对交叉验证，半平面、退化、最小包围圆和覆盖证明均做数值检查。详见 problem_spec.md。

## 7. 新主方案：25点覆盖 + 双侧区间定位

在原有安全PPO之外，仓库新增了基于几何结构改进的新候选 G25O：

- 问题4保证覆盖从31点降到25点，完整覆盖巡航路线由原下界28500m降到18173.85m，至少缩短36.2%；
- 首次有效示向后用成对 no_signal 将距离区间折半，最多12次追加测向加2次末端光学尝试即可保证清除；
- 在本地严格配对中，G25O相对A0在四组场景各100对上平均虚拟时间降低12.64%、3.51%、45.91%、33.87%，800局全部清除。

代码位置：

- `brl/coverage.py::s25_points`
- `brl/bilateral.py`
- `brl/g25o.py`
- `scripts/run_g25o_compare.py`
- `scripts/run_official_g25o.py`
- `docs/NEW_SCHEME.md`

官方G25O 100+100局演练已完成并全部清除：

- 问题3：100/100局，1332/1332源，平均虚拟时间/源319.31s，平均程序运行时间1.351s；
- 问题4：100/100局，1283/1283源，平均虚拟时间/源575.53s，平均程序运行时间3.014s；
- 相对旧安全PPO官方20+20局，问题3、问题4平均虚拟时间/源分别降低25.51%和48.74%；
- 逐局记录见 docs/records/official_g25o_200_runs.csv，汇总见 results/official_g25o_200_summary.json。

G25OR改进版：

- 修正首次示向扇区三角形0.233m外包缺口，改为严格包含半径1500m扇形；
- 用最小包围圆替代包围盒半径作为清除证书；
- 在扫描途中对已获清除证书的源做低绕行插入清除。
- 本地严格配对中G25OR相对A0在四组平均虚拟时间降低24.37%、47.34%、6.02%、33.88%，800局全清。

G25OR官方100+100局也已完成：

- 问题3：100/100全清，1294/1294源，平均虚拟时间/源328.81s，程序运行时间1.462s；
- 问题4：100/100全清，1286/1286源，平均虚拟时间/源556.83s，程序运行时间3.548s；
- 与G25O官方分布差异不显著（p=0.170/0.217），G25OR主要作为严格外包与MEC正确性版本。

官方演练入口：

```powershell
python scripts\run_official_g25o.py --mode 3 --robot-id 202610094088 --base-url http://127.0.0.1:2026
python scripts\run_official_g25o.py --mode 4 --robot-id 202610094088 --base-url http://127.0.0.1:2026
```

## 8. 完整G25O-R实验结论

完整滚动版G25O-R（动态选点+机会测向+Q3动态无源证书+滚动清除）已实现，但本地四组各100对严格配对相对G25OR变化均在±0.2%以内，无实际收益；800局仍全部清除。正式方案继续使用G25OR，G25O-R仅保留为实验分支。

## 9. 三个G25O变体结论

G25O、G25OR、G25O-R full官方各100+100局，全部100%清除。平均虚拟时间/源：P3为319.31/328.81/318.69s，P4为575.53/556.83/558.01s；两两差异均不显著（p>0.17）。正式方案建议G25OR：数学严格外包且成绩等价；G25O最简，G25O-R full暂不采用。

## 10. 已知限制

- 官方模拟器需要登录和联网，本目录无法执行正式测试；本地结果来自严格复现题面规则的 local_env.py 与 mock_server.py，不等同于官方成绩。
- 训练分布是设计者设定的，不是官方案例生成分布。
- 正式提交前应先用演练测试检查程序运行时间、网络重试、日志大小和接口兼容性。
