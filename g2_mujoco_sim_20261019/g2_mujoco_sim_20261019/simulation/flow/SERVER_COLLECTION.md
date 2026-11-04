# 服务器专家数据采集

## 文件与范围

代码和输入均在 g2 仓库内。工作目录为包含 simulation、assets、scenes 的
`g2_mujoco_sim_20261019/g2_mujoco_sim_20261019`。脚本所有相对路径以此为基准。

需上传整个 assets/robot 网格目录、scenes/map9_current.xml，以及：

- simulation/map9.yaml、map9_spread_forward5_full.yaml、plan_map9_vehicle.py；
- simulation/map9_route_preview.py、build_map9_scene.py、full_task_config.py；
- simulation/reference_inputs/map9_scene/scene_map.npz、native_filtered.yaml；
- simulation/flow 全部源码与 data/collection_inputs/ 两个输入文件。

建议同步整个 g2 仓库，避免漏掉间接导入。`data/collection_inputs/` 的
grid.json 与 robot_states.npz 已从原认可结果提取，不需要历史 outputs、
MJB、视频或被忽略的 outputs/debug 地图目录。XML 引用的网格必须保留相对目录。

每次采集只使用一种模式：empty（默认全身姿态，物体固定在货架上），或
loaded（抓取结束姿态，物体随底盘移动）。当前 FM 不以载荷为条件，请
分开训练，勿直接合并两种模式。地图固定，不采集机械臂动作。

## 1. Linux 环境和无规划预检

推荐 Python 3.11，避免与当前 Windows/Python 3.13 的 OMPL 安装问题混淆。
CPU 即可采集，无需 PyTorch 或 GPU。

```bash
python3.11 -m venv .venv-collect
source .venv-collect/bin/activate
python -m pip install -r simulation/flow/requirements-collect.txt
python simulation/flow/generate_experts.py --mode empty --preflight
python simulation/flow/generate_experts.py --mode loaded --preflight
python simulation/flow/test_collection.py
```

预检实际加载网格、栅格和姿态，抽取合法位姿；不调用 OMPL。test_collection
中的规划器为替身，只验证日志、失败保留、断点续采及划分，不验证 OMPL。
OMPL 真实接口必须通过下面的 10 条试采检验。

## 2. 先试采 10 条

```bash
python -u simulation/flow/generate_experts.py \
  --mode empty --motion forward --count 10 --seed 20261103 \
  --planning-seconds 5 --max-attempts 200 \
  --output simulation/flow/data/pilot_empty
```

forward 使用原 Dubins/RRT，只允许前进。`--motion reeds-shepp` 使用原
Reeds–Shepp/RRTConnect，允许倒车，默认仍为前进以便先验证普通 FM。
其最小转弯半径和速度来自现有任务配置，不是实机接口能力证明。

验收时查看 manifest.json：complete=true、accepted=10，划分应为
train=8、validation=1、test=1。每条都有 audit.passed=true 和 SHA256。
所有失败任务保留在 attempts，不能仅按成功数据估计规划成功率。

## 3. 正式采集和续采

```bash
python -u simulation/flow/generate_experts.py \
  --mode empty --motion forward --count 1000 --seed 20261103 \
  --planning-seconds 5 --max-attempts 10000 \
  --output simulation/flow/data/experts_empty
```

改为携物采集，另用输出目录并将每条命令的 mode 保持为 loaded。
10 条试采与 1000 条正式采集分开运行，正式数据自己重新划分，不将
试采清单混入正式测试集。

```bash
python -u simulation/flow/generate_experts.py \
  --mode empty --motion forward --count 1000 --seed 20261103 \
  --planning-seconds 5 --max-attempts 20000 --resume \
  --output simulation/flow/data/experts_empty
```

max-attempts 为累计尝试总预算，允许续采时增大；其他设置必须不变。
程序每次完整尝试后原子更新 manifest，并验证既有文件哈希。Ctrl+C 时
当前未完成尝试会重做，已记录任务不会重复加入。OMPL 内部 RNG 不恢复，
不保证中断前后路径逐位相同，但端点任务按 seed/attempt 可复现。

同一输出目录只允许一个进程，.collection.lock 防止同时写入。kill -9
或断电后可能残留锁；确认没有进程占用后才手动移除该锁再 --resume。
不要在同一目录并行采集。manifest 未登记的临时/孤立 episode 在恢复该
任务时覆盖，不自动当作成功数据。读取数据集时只以 manifest 为准。

完整输入哈希、代码哈希和 MuJoCo/OMPL 版本保存于 settings；续采时
变化会拒绝。运行环境请保留 `pip freeze`，不要在一次数据采集中升级依赖。

## 4. 检查与训练

数据至少保留 64 个状态，dt=0.1 秒，默认拒绝超过 120 秒的路径。
起终位置距离默认 1–12 米。可通过 --min-states、--max-duration、
--min-distance、--max-distance、--dt 调整，并在开始正式采集前冻结。

两种碰撞检查同时执行：栅格 footprint（含 1 cm 余量，未知/越界阻塞），
以及 MuJoCo 全身/携物环境间距（默认 3 mm）；空载还检查固定物体。
时间重采样后，按最多 1 cm、0.02 rad 的线性位姿插值再次检查。
检查通过是离散几何验收，不是连续碰撞、完整自碰撞、动力学或跟踪验收。
状态到动作使用欧拉差分构造，因此残差接近零是构造结果。

```bash
# 在另一个已配置 NumPy/PyTorch 的 FM 环境中执行
python simulation/flow/dataset.py simulation/flow/data/experts_empty
python simulation/flow/train_g2.py \
  --data simulation/flow/data/experts_empty --overfit-windows 0 \
  --device cuda --steps 20000 --output simulation/flow/outputs/fm_empty
```

manifest.json 直接兼容训练加载器，不需要再次转换。划分在切窗口前完成。
数据集仍是原 OMPL 行驶风格，不能视为已覆盖横移/原地旋转。

## 验证状态

本地已通过实际空载/携物预检、栅格阻塞测试、模拟规划失败、恢复续采、
划分、文件损坏拒绝和训练加载兼容测试。当前 Windows 未安装 OMPL，
尚未完成真实 OMPL 的试采；请先运行第 2 步并检查日志，勿直接视为
1000 条采集已验收。若出现 OMPL 接口异常，保留完整 traceback 和环境版本。
