# 普通 FM 后续导航接入

本轮补充代码，尚未运行导航验收。默认 OMPL 完整任务入口保留，现可
通过独立任务配置选择 FM。独立规划入口仍可单独导出参考路线。

## 任务配置

navigation_tasks.json 保存原 G6→G4、G4→G1、G4→G5 的准确起终位姿。
支持自定义 JSON：

```json
{"tasks": [{"name": "route_example", "mode": "empty", "start": [1, 2, 0], "goal": [2, 3, 0.5]}]}
```

mode 为 empty 或 loaded，对应单独的数据集和模型。以上示例坐标只说明
格式，不代表可行任务。全部位置在地图坐标系，角度单位 rad。

```bash
python simulation/flow/plan_navigation_fm.py \
  --tasks simulation/flow/navigation_tasks.json \
  --empty-checkpoint simulation/flow/outputs/fm_empty/best.pt \
  --loaded-checkpoint simulation/flow/outputs/fm_loaded/best.pt \
  --steps 100 --candidates 16 --device auto \
  --output simulation/flow/outputs/navigation_fm
```

该入口需要 FM 和采集场景两套依赖（Torch/NumPy/MuJoCo/SciPy/PyYAML），
不需要 OMPL。FM 在每条任务上直接生成固定时域的候选，独立检查动作
边界、端点、运动学残差、占据栅格与三维几何；合格候选按路径长度选择。
这属于普通 FM + 拒绝筛选，不使用 CBF/CLF 或其他生成过程修正。

position-tolerance 默认每步 0.02 m，angle-tolerance 默认每步 0.02 rad；
只是可配置实验阈值，不能视为实机能力证明。所有候选的诊断均写入
navigation_report.json，没有合格候选时报告失败，不回退 OMPL。通过
离线参考检查也不会标为硬件执行通过。

## 固定时域限制

64 状态、dt=0.1 对应 6.3 秒，不足以覆盖目前全部原始导航任务。
若起终距离大于速度上限乘以时域，入口直接报错。正式长路线模型需
用足够长的 horizon 重新训练，例如 384 状态对应 38.3 秒（仍需检查
具体任务、绕障和航向变化是否在时域内）。

```bash
python simulation/flow/train_g2.py --data simulation/flow/data/experts_empty \
  --horizon 384 --overfit-windows 0 --device cuda --steps 20000 \
  --output simulation/flow/outputs/fm_empty_long
```

采集时也应设置 --min-states 384，保证各划分有可用完整窗口。选择更长
horizon 会增加显存和计算量。不要把模型原来的 6.3 秒输出随意拉伸到
更长时间，也不要在不保留状态/动作一致性的情况下拼接长路径。

## 输出与播放

每条成功路线含 poses/states、actions、timestamps、dt、loaded，以及
兼容原格式的 ds/dyaw/longitudinal/lateral。保存完整预测动作，不用
位置差分替换模型动作。失败任务不输出新的路线文件。

render_map9_full_task.py 的导航时间读取已适配：存在 timestamps 时直接
使用相邻物理时间差；旧 OMPL 文件继续按距离、速度和换向停顿计时。

独立规划目录只有路线和报告，不包含完整任务的 pickup/placement/scene/state。

## 完整任务的 FM 配置入口

`simulation/map9_flow_full.yaml` 继承原任务参数，将 planner 设为 flow，
并使用独立输出 `simulation/outputs/map9_flow_full`。修改 vehicle.flow
中的空载、携物检查点路径后运行：

```bash
python simulation/run_spread_forward5_full.py --task simulation/map9_flow_full.yaml --skip-render
```

不加 --skip-render 则在验证通过后录制视频。配置支持 extends，父配置
路径相对于项目根目录。完整任务会准备机械臂与抓放数据，调用统一的
plan_base_trajectory，并保留原货架退出位姿分段策略。每段 FM 都采用
模型固定时域；拼接仅允许端点吻合、dt 一致的结果，保留两段预测动作，
航向角按等价 2pi 分支连续拼接，不进行时间拉伸。

FM 失败时终止，无 OMPL 回退；阶段失败记录 execution_failure.json。
验证器重新读取路径并计算物理时间、端点、动作边界和离散运动学残差。
FM 导航不沿用 OMPL 无横移和曲率判据；局部抓放仍保留原独立审核，
全身/物体环境几何检查保持运行。通过仅代表离线参考，不是实机验收。

完整任务仍需要原场景构建和机械臂依赖，以及 map9.yaml snapshot 指向
的完整地图元数据等历史输入；采集环境的最小输入包不能代替全部原
仿真依赖。准备阶段仍沿用原共享抓姿目录。此轮未执行完整任务。
