# 普通 FM：服务器运行指南

所有项目代码均在 g2 仓库内；不需要上传 `sad-flower-main`。
这是本地 Transformer + 线性 CFM 基线，不是论文网络的逐项复刻。
普通 FM 不修正动作、不求解 QP、不启用 CBF/CLF，也不拒绝不安全候选。

## 1. 目录与环境

在服务器进入包含 `simulation`、`assets`、`scenes` 的项目目录：

```bash
cd g2/g2_mujoco_sim_20261019/g2_mujoco_sim_20261019
python3 -m venv .venv-fm
source .venv-fm/bin/activate
python -m pip install -r simulation/flow/requirements-fm.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

推荐 Python 3.11 或更新版本。GPU 训练需安装与服务器驱动匹配的 PyTorch
发行包；上面 pip 默认选择不保证 CUDA 可用。选择 `--device cuda` 会在
CUDA 不可用时报错，`--device auto` 会自动退回 CPU。训练无需 MuJoCo、
OMPL、Gym、D4RL 或 PyYAML。不要安装原始整包 requirements 来运行纯 FM。

需同步 `simulation/flow/data/converted/` 下的三个 NPZ 及
`dataset_manifest.json`。Git 忽略 outputs，所以本地模型不会自动上传；
服务器可重新训练，也可单独上传 `latest.pt`。独立采样不需要训练集。

## 2. 数据与自动检查

```bash
python simulation/flow/dataset.py simulation/flow/data/converted
python simulation/flow/test_fm.py
```

预期窗口数为 train 30、validation 26、test 6，张量通道为
`[vx, vy, omega, x, y, theta]`。训练集统计独立用于归一化。
正式数据可使用 `dataset_manifest.json` 或生成器的 `manifest.json`；
每个 episode 必须有 file/split/sha256，NPZ 包含 states、actions、timestamps、
dt、loaded。states 比 actions 多一行。完整 episode 先划分再切窗口。

## 3. 先执行两步短训练

```bash
OMP_NUM_THREADS=4 python simulation/flow/train_g2.py \
  --device auto --steps 2 --log-every 1 --width 32 --depth 1 \
  --output simulation/flow/outputs/server_smoke
```

检查 metrics.jsonl、latest.pt、best.pt、sample_report.json 是否生成。
每次使用新的 output 路径，脚本拒绝覆盖非空目录。

## 4. 小样本训练

```bash
OMP_NUM_THREADS=4 python simulation/flow/train_g2.py \
  --device auto --steps 2000 --width 64 --depth 2 --overfit-windows 4 \
  --output simulation/flow/outputs/server_overfit
```

先观察固定噪声 train_probe 是否下降。validation 变差是过拟合信号。
三条路线只用于检查训练链路，不能支持泛化结论。test 当前为空载路线，
train/validation 为携物路线；载荷状态尚不是模型条件。正式实验应先
扩充数据，并固定单一载荷模式，或另外实现载荷条件。

扩充数据后使用 `--overfit-windows 0 --steps 20000` 训练全部训练窗口。
`--data` 指定新数据目录。`--horizon` 是状态点数，固定物理时域为
`(horizon-1)*dt`。默认 64、0.1 秒对应 6.3 秒；不能把此模型直接用于
任意长度的完整运输路线。更长路线应重新选时域训练或另写分段策略。

继续训练（总目标步数 4000，保持网络和数据选择参数一致）：

```bash
python simulation/flow/train_g2.py --device auto --width 64 --depth 2 \
  --overfit-windows 4 --steps 4000 \
  --resume simulation/flow/outputs/server_overfit/latest.pt \
  --output simulation/flow/outputs/server_resume
```

保存模型与优化器状态；继续训练不恢复 DataLoader 游标和全部随机状态，
不是逐比特等价续跑。每次日志时保存 latest，验证损失更低时保存 best；
latest 写入采用临时文件替换。训练日志实时输出，可由服务器终端重定向。

## 5. 独立位姿条件采样

以下坐标仅示范命令，不代表地图中可行任务。theta 用弧度，输入为连续
航向角分支；不要在 ±pi 两侧未经展开直接指定目标角。

```bash
python simulation/flow/sample_fm.py \
  --checkpoint simulation/flow/outputs/server_overfit/latest.pt \
  --start 1 2 0 --goal 2 3 0.5 --candidates 4 --steps 100 \
  --device auto --loaded --output simulation/flow/outputs/server_samples
```

只输入起终位姿；dt/网络/归一化取自 checkpoint。loaded 仅记录元数据，
不影响模型。候选 NPZ 同时包含 states/actions/timestamps 和旧 route 的
poses/ds/dyaw/longitudinal/lateral，便于后续适配；输出不自动进入 g2
导航流程。播放若读取旧距离计时逻辑，还需改为使用 timestamps。

## 6. 批量评估保留集

```bash
python simulation/flow/evaluate_fm.py \
  --checkpoint simulation/flow/outputs/server_overfit/best.pt \
  --split validation --max-windows 10 --candidates 4 --steps 100 \
  --device auto --output simulation/flow/outputs/server_eval
```

测试集替换为 `--split test`。评估严格检查划分清单与训练 checkpoint 一致。
所有候选都计入结果，不通过挑选最好候选掩盖失败；不计算碰撞成功率。
evaluation.json 包含端点误差、位置/角度残差、动作超限、路径长度和参考
轨迹误差。参考误差只是诊断，另一条不同形状的路径也可能有效。

## 当前验收范围

普通 FM 数据→训练→保存→加载→采样→数值评估可以独立运行。
尚未提供普通 FM 生成轨迹的地图碰撞验收、实机执行或完整路线成功率。
已有 check_dataset.py 用于原始转换路线，不能直接等同于对新模型输出
的验收。任何 FM 候选均需进一步检查，禁止将 accepted_for_execution=false
的结果当作执行通过。GPU 路径需由服务器验证，本地仅测试 CPU。
