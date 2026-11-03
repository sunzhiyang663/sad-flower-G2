## 当前：左托点前移 5 cm 的地图 9 完整流程（10月13日）

新输出独立保存于 `simulation/outputs/map9_spread_forward5_full`，保留已认可的旧抓姿和原状态机配置。左手相对壳体沿长边前移 **5 cm**；为避免放置时腕部相机贴近桌面下沿，另下调 **3 mm**。右手朝向、手指上下分散接触和抓取站位保持认可版；抓取时右肘低于肩 10 cm。左手图见 [托抓预览](../../simulation/outputs/grasp_spread_forward5/paired_grasp_fullbody.png)。

流程：默认全身姿态从 G6 导航至 G4 → 假定相机观测 → 降腰、在货架外依次预摆右臂和左臂 → 底盘直进 40 cm → 右手闭合并先拖出 19 cm → 左手接入包握 → 双手再退出 6 cm、抬升 10 cm → 持物直退 35 cm 后按车辆曲线驶往 G1/G5 → 保持腰部，双臂稍收近并下降 → 自由前端先搭上桌沿 1 cm → 左手松开并撤离 → 右手沿台面先推送、底盘跟进 → 右手松开撤回 → 底盘直退 50 cm → 手指、双臂、腰颈复位。所有机械臂轨迹均为规划结果。

下降时双手共同后收 4 cm、左移 2 cm；这是机械臂持物调整，底盘没有横移。松左手后，右臂先向前推 6 cm、向右调整 4 cm，再由底盘直进约 22.5 cm，避免躯干挤向桌沿。腰颈从预摆完成直到放置撤离都保持固定，仅起始降高和最终复位改变。

- [G6 → G4 → G1 完整 MuJoCo 视频](../../simulation/outputs/map9_spread_forward5_full/G6_G4_G1_full_task.mp4)
- [G6 → G4 → G5 完整 MuJoCo 视频](../../simulation/outputs/map9_spread_forward5_full/G6_G4_G5_full_task.mp4)
- [抓取与放置双视角近景](../../simulation/outputs/map9_spread_forward5_full/grasp_front_land_closeup.mp4)
- [整段验证](../../simulation/outputs/map9_spread_forward5_full/validation.json)、[加密局部检查](../../simulation/outputs/map9_spread_forward5_full/local_motion_audit.json)、[顺序和复位检查](../../simulation/outputs/map9_spread_forward5_full/preposition_audit.json)

三段视频均已完成编码并核对：G1 116.5秒、G5 113秒、近景58.5秒，H.264 / 960×540 / 8fps。两条完整视频末尾均包含放置、退离和全身复位；`completion.json` 保存帧数及文件哈希。软件渲染关闭了多重采样抗锯齿，不改变碰撞几何。

验证通过：局部轨迹按最大 0.0075 rad / 2.5 mm 加密，共 4,329 状态；两条地图流程分别 2,569 / 2,452 状态，机器人及壳体环境碰撞检查零失败。最终壳体底面为 75 cm，最终关节恢复默认姿态。车辆路径使用 Reeds–Shepp / Dubins，假定最小转弯半径 0.6 m，只有前进、倒车和曲线转向，无蟹行。空载路径 3.668 m，携物路径 G1/G5 为 11.562/9.244 m。

22 组理想前后激光扫描中，壳体产生 11,693 个回波；按壳体位姿过滤后漏滤 0、环境误删 0，过滤前后碰撞几何哈希不变。地图沿用原离线清理策略（去除 10,913 个高度带外/非占据列体素），另清理动作范围内 8 个扫描体素；原地图、占据栅格和显式家具模型保留。这不是实机 GDK 闭环验收，未部署过滤器。视觉观测、底盘高度、桌面相对点位仍为离线假设，抓放采用运动学附着，未验证触觉、摩擦、承载动力学或完整手内自碰撞。

复现全流程（项目根目录，纯离线，不连接机器人；会重算并录制视频）：

```bash
.venv-mujoco/bin/python simulation/run_spread_forward5_full.py
```

参数位于 `simulation/grasp_spread_forward5.yaml`（左托点调整）和 `simulation/map9_spread_forward5_full.yaml`（全流程）。单独运行地图脚本时设置 `G2_SIM_TASK=simulation/map9_spread_forward5_full.yaml`，未设置时也默认使用本轮配置。新托点未替换实机或 `spread_grasp` 短段任务参数。

## 结果清理（10月13日）

按用户要求，旧仿真视频、截图、轨迹和试算输出已删除。`simulation/outputs` 仅保留本轮 `map9_spread_forward5_full` 和 `grasp_spread_forward5` 两个目录；本轮所有文件清理前后 SHA-256 一致。

复现所需的基础地图场景、货架定义、抓姿初值和比较基线已缩减为 15 个文件，迁至 `simulation/reference_inputs`，不再依赖已删除目录。实机视觉和原 `spread_grasp` 状态机仍采用之前认可的抓姿数值；仅更新源文件路径，不替换成左托点前移版本。视觉抓姿原始三文件的哈希保持不变。

迁移关系见 [输入清单](../../simulation/reference_inputs/manifest.json)，删除清单及保留结果哈希见 [清理记录](../../simulation/cleanup_manifest.json)。原生 GDK 缺失插件清单保留于 `simulation/reference_inputs/map9_scene/missing_plugins.json`。原始地图、URDF、复现代码及实机采集记录保留。旧试验脚本保留供代码参考，已不再保留其输出或保证其历史独立入口可运行；当前完整复现入口为上面的 `run_spread_forward5_full.py`。
