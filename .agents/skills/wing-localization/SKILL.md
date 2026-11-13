---
name: wing-localization
description: 在本项目中操作或维护机翼内 AprilTag 与三路红外定位，标定相机和标签布局，复现动捕对照实验，或按相机快照重算位置、姿态和红外误差。适用于 11.11 第三次测试的快照重算、异常组筛选和标签偏移验证。
---

# 机翼内视觉与红外定位

从项目根目录运行 `python positioning.py`。实时实现位于 `localization/`，配置位于 `config/`，下文的 PowerShell 命令从项目根目录运行。先确定用户在做实时采集、设备标定、离线分析还是代码维护。`experiments/` 下带日期的目录是独立实验批次，组号只在本批次内有意义。

## 按任务选择入口

| 任务 | 入口 |
|---|---|
| 查看界面、连接相机与串口、采集联合样本 | `positioning.py`，见[实时采集与标定](references/acquisition-and-calibration.md) |
| 修改位姿解算、传感器几何、平面拟合或记录格式 | `localization/positioning_core.py`，优先复用现有的 `VisionEstimator`、`fit_plane`、`record_snapshot`、`save_record` |
| 修改设备连接、线程、窗口或快捷键 | `localization/positioning.py` 的 `Acquisition` 与 `make_window` |
| 检查对焦、拟合固定内参或多焦点内参 | `calibration/focus_probe.py`、`calibration/camera_calibration.py`、`calibration/autofocus_calibration.py`，见[实时采集与标定](references/acquisition-and-calibration.md) |
| 重建动捕参考、计算误差、处理异常组或拟合标签偏移 | 见[实验复现与误差分析](references/experiment-analysis.md)，先选对批次和输出目录 |

根目录的 [README](../../../README.md) 详列窗口交互、协议和 CSV 字段。`legacy/visual_positioning_old.py` 与 `legacy/infrared_positioning_old.py` 供历史对照。配置优先读实际 JSON，实验参数优先读该批的记录、说明和分析脚本。

`calibration/` 保存标定工具、会话、拟合参数和诊断。`output/acquisition/` 放新采集的 CSV 和相机快照，`output/` 也存放棋盘格和汇报材料。`.verification/` 放界面及标定检查产物。每批实验的原始记录、回放与误差统计可从该批目录查找。

## 坐标与单位

- `T_A_B` 将 B 系坐标转换到 A 系。W 是机翼系，R 是机器人前端盖中心，C 是 OpenCV 相机光心坐标系。R、C 的约定为 X 向右、Y 向下、Z 向前。
- 在线配置、CSV 变换矩阵和核心几何用米。动捕分析的 `*_mm` 矩阵、CAD 标记点和误差用毫米。旋转矩阵无单位，姿态误差用相对旋转角度（°）。
- `T_W_R = T_W_C @ inverse(T_R_C)`。动捕相机真值为 `T_W_C_gt = T_W_R_gt @ T_R_C`。当前安装外参旋转为单位阵，相机光心在 R 系中为 `(0, -29.5, 4.0) mm`。历史批次以其配准脚本和记录中的外参为准。
- `config/tag_layout.json` 的每个键是标签 ID，值为 `T_W_Tag`。左上角是旋转，最后一列前三项是标签中心，单位为米。实时配置为 `tag25h9`、边长 `0.012 m`，包含 1–34 号标签。11.11 第三批的回放布局见该批目录的 `replay_tag_layout.json`。修改后核对标签 ID、单位、轴向和刚体矩阵。
- 在线 STL 默认按 `0.001` 将毫米缩放到米，再应用各自的 CAD 配准矩阵。离线可视化按分析脚本的单位处理。相机标定、安装外参、标签布局、STL 配准分别对应不同的物理量。

## 数据与结果

- 保存按 S 时界面显示的样本。数据组号可重复。复现时用 `(group_id, snapshot_at)` 关联同帧 CSV、快照 JSON 和同名 PNG，保留原始采集记录与 MATLAB 文件。
- 保存的 PNG 已按采集方向处理并绘制标注。带逐帧元数据的回放采用 CSV 中的原始角点、JSON 中的检测中心与 `camera_matrix`、`dist_coeffs`，交给当前 PnP。旧批次缺少这些字段时，注明实际可用的点和内参来源。
- 精度评估使用独立动捕参考。分别报告相机光心距离误差、相对旋转角和有效红外组；视觉像素残差与动捕误差分开读。先核对配对、单位、有效组数和筛选依据，再比较修正前后结果。
- 回放和标定试验写入对应批次的结果目录。复用已有结果前查 `input_manifest.json`，尤其检查布局、配置和帧元数据是否属于同一基线。估计出的偏移是待验证的校准参数，按用户指定的范围决定是否写入工作配置。

## 验证与交付

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

按变更范围选择测试。`tests/test_positioning.py` 覆盖几何、PnP、红外平面与保存，设备连接和相机标定有独立测试。涉及 Qt/PyVista 交互时可运行 `python tests/gui_smoke.py`，需要桌面和可用的 OpenGL。离线重算还应核对各统计方案的组号、均值/RMSE、生成文件与报告是否一致。交付时注明用到的配置、数据批次、剔除或替代的组、关键数值与结果文件。
