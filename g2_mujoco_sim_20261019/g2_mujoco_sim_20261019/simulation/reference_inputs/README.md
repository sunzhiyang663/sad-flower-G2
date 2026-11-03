# 仿真必要输入

旧仿真输出清理后保留的最小数据集合；无旧视频或截图。manifest.json 记录原路径、新路径和原始 SHA-256。

- map9_scene：完整流程使用的基础场景、地图体素、初态与 footprint 参数；missing_plugins.json 为原生 GDK 插件缺失清单。native_filtered.yaml 在当前离线流程中只读取 footprint，不代表已部署导航。
- rack / extraction：货架与机器人建模参数、抽取参考。
- grasp_seed：加载 Checker 的历史抓姿起点。
- travel_baseline：当前运动审计的关节行程比较数据。
- accepted_grasp：已认可抓姿，供本轮左托点调整、原短段状态机和实机视觉抓姿导出使用。

当前视频和验证报告在 ../outputs/map9_spread_forward5_full 与 ../outputs/grasp_spread_forward5。
