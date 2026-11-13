# 实时采集与标定

以下命令从项目根目录执行。定位、标定工具和离线分析的依赖清单分别是 `requirements.txt`、`requirements-calibration.txt` 和 `experiments/26.11.1红外+相机精度测试/requirements.txt`。第三次测试的快照重算需要定位依赖和离线分析依赖。按任务安装所需的清单。

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-calibration.txt
python -m pip install -r "experiments/26.11.1红外+相机精度测试/requirements.txt"
```

## 启动、配置与保存

```powershell
python positioning.py --demo --csv "output/demo_records.csv"
python positioning.py --list-ports
python positioning.py --camera 1 --camera-backend dshow --port COM4 --baud 115200
```

- 模拟模式用于验证窗口、保存和显示，记录带 `demo=True`。需要隔离试采的 CSV 与 PNG 时，复制一份配置并为 `csv_file` 与 `snapshot_dir` 指定测试路径。单独传 `--csv` 只覆盖 CSV 路径。
- `--config` 指定配置文件。配置内的相对路径从配置文件所在目录解析；命令行覆盖的 `--csv`、`--focus-map` 从当前目录解析。将配置复制到其他目录时同步检查所有引用路径。
- 默认相机索引为 1、分辨率 640×480、15 FPS、旋转 180°。相机窗口可重连、切换、手动调焦和恢复自动调焦。换设备时核对分辨率、方向、相机内参和安装外参。
- 串口默认 COM4、115200，可按 CH340 的 `1A86:7523` 识别设备。显式 `--port` 指定端口。红外和视觉独立采集，保存时结合数据年龄、接收时间和状态判断有效性。对焦驱动回读及软件保存时间可用于配对和诊断，逐曝光硬件同步需要另有记录。
- S 打开组号输入框，保存触发时冻结的显示样本。Space 暂停或恢复显示，R 重置视角，Q 退出。记录可能只有 CSV，也可能附带 PNG；动态内参模式下有相机帧和上下文时，还写同名 JSON。相机、红外或外参失效时，对应位姿和平面字段留空。

## 相机内参与对焦

| 需求 | 入口与读法 |
|---|---|
| 判断驱动可否读写焦点、回位是否可重复 | `python -m calibration.focus_probe capture --backend dshow`。用 `python -m calibration.focus_probe analyze` 复查保存的会话，见[对焦检查说明](../../../../calibration/对焦检查说明.md)。 |
| 固定焦点内参 | `python -m calibration.camera_calibration board`、`capture`、`calibrate`、`undistort`。按[内参标定说明](../../../../calibration/README.md)选择实物格长、焦点和采集目录，检查逐图 RMSE 与去畸变图。 |
| 多焦点内参 | `python -m calibration.autofocus_calibration capture` 与 `fit --session`。见[多焦点标定说明](../../../../calibration/多焦点自动标定说明.md)，检查训练姿态、独立验证图、端点和插值中点。 |

棋盘格参数的 9×6 指内角点，默认实物单格 20 mm。相机标定与定位须用同一分辨率及同一图像方向。已旋转的采集图和快照直接按其保存方向读取。固定内参模式将测试配置的 `focus_map_file` 设为 `null`，由 `calibration_file` 指向实际内参。当前固定内参为 `calibration/camera_intrinsics_refined.json`，旧 `calibration/charuco_calib.json` 可用于有明确要求的历史对比。

实时动态内参由 `localization/focus_map.py` 处理。稳定的驱动读数在标定档位之间线性插值，超出拟合档位时用同侧最近两档线性外推。当前表的独立验证区间为 272–304，拟合档位与通过验证的区间分别看待。报告逐帧回放时优先使用该帧实际记录的完整 K/D；对焦值高于拟合档位或落在未通过验证的范围时注明。

## 视觉求解和红外平面

- `VisionEstimator` 在已知标签的局部四角上初始化单标签 PnP，多标签使用角点 RANSAC。检测中心直接取 `tag.center`，对应标签坐标原点。进入优化前检查同标签角点支持和中心投影偏差，当前中心权重为单角点的 4 倍，多标签的图像中心权重由配置控制。保留角点顺序和 `inlier_indices_json` 的四角索引约定。
- `reprojection_rmse_px` 是实际参与点的未加权像素 RMSE，`weighted_reprojection_rmse_px` 是优化目标对应的加权版本。像素残差可用于诊断，单个平面标签的两支姿态可能同时给出低残差。
- `config/infrared_calibration.json` 定义 S1–S3 的线性标定。传感器位置和方向取 `localization/positioning_core.py` 中的 `SENSOR_POSITIONS`、`SENSOR_DIRECTIONS`。计算测点时保持串口编号、系数与安装几何对应。
- `fit_plane` 需要三路有效正距离、接收时间满足 `sensor_max_skew_s`（当前 0.25 s）、三点不共线。机器人系平面写作 `n_R · X_R + d_R = 0`，法向量朝向机器人原点，距离取原点至平面的垂直距离。
- 需要验证或重拟合红外系数时，先看[11.1 红外重新标定报告](../../../../experiments/26.11.1红外+相机精度测试/results/infrared_calibration/红外重新标定报告.md)。该批的离线系数及留一验证由 `calibrate_infrared.py` 生成，工作配置中的系数由 `config/infrared_calibration.json` 决定。
