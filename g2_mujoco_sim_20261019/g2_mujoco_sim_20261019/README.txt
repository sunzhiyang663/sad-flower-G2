G2 MuJoCo 仿真交付包 — 2026-10-19

本包对应已认可的 map9_spread_forward5_full：左托点沿壳体长轴前移5cm、下调3mm。
完整流程：G6默认姿态出发 → G4识别（假定相机位姿）→ 降腰/远处预摆双臂 → 右手先拖出 → 左手接入包握 → 双手抬升 → 车辆式导航到G1/G5 → 前端搭台/松左/右推 → 退离并复位。

1. 包含内容
- scenes/map9_current.xml：可编辑的完整MJCF场景，含机器人、货架、75cm桌面和清理后的点云碰撞体；相对引用包内网格。
- simulation/outputs/map9_spread_forward5_full/scene.mjb：原始已验证的编译场景。其state.npz、pickup.npz、placement.npz和route_*.npz是回放必需的数据。
- simulation/outputs/：仅本轮完整流程和左手前移5cm的局部抓取结果，含MP4、PNG、轨迹、审计报告。均保留原字节。
- assets/robot/：G2_t2_crsB_o12_t2 URDF、SRDF与76个实际引用网格；assets/sensors包含当前前后激光外参快照；assets/grasp_kinematics_constants.json为抓姿参考。
- simulation/*.py、*.yaml：本轮求解、导航、碰撞检查和录像所需源码及配置。部分共享模块含历史函数，交付支持入口以本说明为准。
- simulation/reference_inputs/：复现需要的精简参考输入；outputs/debug/map9_view/是地图9的输入快照。
- config/manipulation.yaml：默认关节配置；config/task/spread_grasp.yaml是原认可短段状态机参数参考，未替换成前移5cm版本。本包未提供实机执行入口。
- requirements.txt：本机Python依赖版本；MANIFEST.sha256：交付文件校验。

2. 环境准备（推荐Linux，已验证主机为Ubuntu22.04/aarch64/Python3.10）
在解压目录下运行：
  python3.10 -m venv .venv-mujoco
  .venv-mujoco/bin/python -m pip install -r requirements.txt
Linux录制需要ffmpeg和OpenGL/OSMesa，例如Ubuntu：
  sudo apt-get install ffmpeg libosmesa6 libglfw3
字体随包提供。ARM版OSMesa辅助库在simulation/vendor_libs中，仅ARM使用；x86_64使用本机安装的库。不复制原虚拟环境。其他架构/系统的依赖安装未在本机验证。
MuJoCo版本为3.14.0；MJB是编译格式，应使用相同版本。需要修改模型或迁移版本时使用MJCF/URDF重新编译，并重新验证轨迹。

3. 先检查交付文件（不连接机器人）
  .venv-mujoco/bin/python tools/check_package.py
可选单帧无窗口渲染检查：
  .venv-mujoco/bin/python tools/render_check.py

4. MuJoCo交互式回放（有图形桌面）
  .venv-mujoco/bin/python tools/replay.py --goal 1
  .venv-mujoco/bin/python tools/replay.py --goal 5
可加 --speed 0.5；循环播放，关闭窗口结束。仅加载保存的qpos/底盘位姿/壳体附着变换并调用mj_forward。
单看可编辑场景：
  .venv-mujoco/bin/python -m mujoco.viewer --mjcf=scenes/map9_current.xml
默认viewer中请启用几何显示组3/4（壳体/地图），关闭组0（碰撞包络）；tools/replay.py已自动设置。
单独XML不含回放时序，直接点击物理运行不等于抓取控制。使用tools/replay.py看完整运动。

5. 重新规划并录制（会更新当前输出，先复制一份解压目录用于新实验）
  .venv-mujoco/bin/python simulation/run_spread_forward5_full.py
只按已有轨迹录制：
  .venv-mujoco/bin/python simulation/render_map9_full_task.py --goal 1
  .venv-mujoco/bin/python simulation/render_map9_full_task.py --goal 5
  .venv-mujoco/bin/python simulation/render_map9_grasp_closeup.py
无图形界面默认使用OSMesa；有可用EGL驱动时可在命令前设置MUJOCO_GL=egl。
主要参数：simulation/map9_spread_forward5_full.yaml（任务/导航/放置/录像）、simulation/grasp_spread_forward5.yaml（托点）、simulation/map9.yaml（地图/家具/点云/激光）。
模型和雷达路径均已改为包内相对路径，原脚本请在包根目录运行。

6. XML导出与场景修改
  .venv-mujoco/bin/python tools/export_scene_xml.py
导出器用原场景生成代码恢复拓扑，再导出当前MJB的几何/碰撞属性；检查记录在scenes/export_check.json。XML存在浮点文本舍入，原MJB仍为保存结果的基准。
导出器要求当前MJB与基础生成器拓扑相同。若新增/删除物体，请在新实验中重新生成场景、轨迹并验证，不能将旧轨迹视为已经验证。
修改XML不会自动影响原流水线：原流水线读取reference_inputs/map9_scene/scene.mjb并重新进行点云清理。如需让流水线使用新场景，请同步修改生成代码/配置并更新基础模型、state等数据；单纯探索新场景可以直接从XML开始。

7. 验证范围
这是运动学与几何仿真，非原生GDK闭环导航。抓握使用位置目标和刚性附着，无触觉、摩擦、抓稳和负载动力学保证；未验证完整手内自碰撞。壳体从模拟感知点云滤除后仍保留碰撞体。原生导航插件缺失清单在reference_inputs/map9_scene/missing_plugins.json（完整路径前缀为simulation/）。
未打包GDK服务/实机接口、机器人账号信息、实时CV采集记录和整个开发环境。仅相应离线输入快照；历史报告中的原机路径仅为来源记录。
