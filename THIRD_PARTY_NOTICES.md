# 依赖、图片与代码来源

- 原模型由 Ultralytics YOLO11 训练，权重和ONNX元数据标注 AGPL-3.0，来源 [Ultralytics](https://github.com/ultralytics/ultralytics)。版本、原始元数据和SHA见模型资料。
- `third_party/rdk_x5_export_monkey_patch.py` 原样来自 D-Robotics/rdk_model_zoo，commit `4f378d74fcf3507ba1336a14f5bf66bfecab2990`；源文件保留Apache2.0声明，LICENSE见同目录。
- 演示与量化图片来自用户本地的 Roboflow gauge-key v1 导出；其原始说明标注CC BY4.0，提供者与数据集页面：[2446813694's workspace / gauge-key](https://universe.roboflow.com/2446813694s-workspace-l6ozi/gauge-key)。原始README随附于samples目录；图片原字节保留。
- `tests/fixtures` 为历史用户提供的几何回归图片，来源和用途按该目录说明保留，不纳入本次独立精度或现场验收。
- 方法参考：[用户指定的视觉仪表文章](https://blog.csdn.net/weixin_53354122/article/details/150534544)。本仓库不转载文章正文；角度换算使用项目自己的atan2和分段刻度插值实现。

本交接包含项目源代码与所列第三方文件，不将所有文件统一改标为另一种许可证。
