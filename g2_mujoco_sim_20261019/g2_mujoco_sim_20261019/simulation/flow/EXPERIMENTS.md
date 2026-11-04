# 普通 FM 批量实验和绘图

本轮只编写代码并检查语法；未运行批量实验、数值验收或图片渲染。
以下命令均在包含 simulation 的项目根目录运行。

## 安装与配置

```bash
python -m pip install -r simulation/flow/requirements-experiments.txt
```

requirements 增加 Matplotlib。GPU 版 PyTorch 仍按服务器驱动选择。
修改 experiments_example.json 中的 data、seeds、device、train、evaluate。
默认示例使用三种子、空载正式数据、全部训练窗口和 validation/test 评估。
空载与携物分别建立实验配置；普通 FM 尚未以载荷作为条件。
长时域任务需同步修改 train.horizon，并确认各数据划分包含足够长轨迹。

## 批量执行

```bash
python simulation/flow/run_experiments.py \
  --config simulation/flow/experiments_example.json \
  --output simulation/flow/outputs/experiments_empty --dry-run

python -u simulation/flow/run_experiments.py \
  --config simulation/flow/experiments_example.json \
  --output simulation/flow/outputs/experiments_empty
```

dry-run 仅读取配置/数据清单并显示指纹，不启动子进程或创建结果目录。
真实运行顺序为每个 seed 的训练→最佳验证模型评估→可选损失曲线。
所有 seed 完成后生成汇总；某个 seed 失败，继续其他 seed 并保留失败日志。
实验依次运行，避免同时争用 GPU。plots=false 可跳过绘图。

```bash
python -u simulation/flow/run_experiments.py \
  --config simulation/flow/experiments_example.json \
  --output simulation/flow/outputs/experiments_empty --resume
```

resume 跳过已完成且产物哈希一致的阶段；失败或中断阶段使用新的
attempt 目录重做，保留旧日志。这里的阶段重试是重新训练，不等同于
train_g2.py 的检查点续训。已完成产物损坏时拒绝跳过，需要另建输出。
配置、数据划分或代码变化也要求另建实验目录。

experiments.json 记录每阶段命令、状态、耗时、日志和关键产物哈希。
.experiment.lock 防止同一目录并行写入；异常断电后确认无进程运行再
移除残留锁。日志写在各阶段 attempt_NNN.log 中，可用 tail -f 查看。

## 单独绘图

```bash
python simulation/flow/plot_results.py \
  --run simulation/flow/outputs/overfit_run1 \
  --output simulation/flow/outputs/figures_overfit

python simulation/flow/plot_results.py \
  --trajectory simulation/flow/outputs/server_eval/window_0000_candidate_000.npz \
  --reference simulation/flow/data/converted/route_G4_G5.npz --window-start 0 \
  --output simulation/flow/outputs/figures_trajectory
```

第二条的 reference/window-start 应与 evaluation.json 候选来源对应，
示例路径须替换为实际结果。输出 loss.png、trajectory.png、
controls_residual.png；只生成输入参数对应的图。采用 Agg，无需显示器。
轨迹图不显示地图障碍，因此不能凭图片判断无碰撞。

## 汇总结果

```bash
python simulation/flow/summarize_results.py \
  --input simulation/flow/outputs/experiments_empty \
  --output simulation/flow/outputs/experiments_empty/summary
```

summary.json 包含逐评估文件统计、实验阶段状态与失败日志，以及输入
目录中 navigation_report.json/validation.json 的结果；evaluations.csv
便于后续分析。不将不同模型、种子或数据划分混成一个成功率。

generated_candidates 为实际写出的候选数；requested_candidates 包括
生成失败窗口的候选预算。数值通过率以全部请求候选为分母，要求端点、
动作边界、位置/角度残差同时满足阈值。它不是碰撞或执行通过率。
碰撞率和实机成功率未测量时为 null，不把 false/缺失误记为成功。
重叠窗口仍相关；evaluate.stride=64 仅减少窗口重叠，尾窗口仍可能重叠。

评估计时在 CUDA 上前后同步，记录每个窗口整批候选的采样时间及摊销
每候选耗时，不含模型加载/文件写入。首次调用包含冷启动开销；摊销
时间不是独立单候选延迟。参考轨迹误差只作诊断，不等同路径有效性。
