# 相机内参重新标定

从项目根目录运行 `python -m calibration.camera_calibration`。采用 OpenCV 普通针孔模型和 5 个畸变系数 `[k1, k2, p1, p2, k3]`，与当前 AprilTag 定位程序兼容。相机实际的内参需要拍摄棋盘格后计算；随附文件不含任何伪造的标定结果。

## 1. 打印棋盘格

已经生成：

- `output/pdf/chessboard_9x6_20mm_A4.pdf`：优先打印此矢量 PDF，A4 横向。
- `calibration/boards/chessboard_9x6_20mm_A4.png`：254 DPI 的 PNG，2970×2100 像素。

棋盘包含 **10 列×7 行方格，9 列×6 行内角点，每格 20 mm**，黑白区域为 200×140 mm，外围保留白边。程序里的 `--cols`、`--rows` 始终指内角点数量。

打印选择“实际大小 / 100%”，关闭“适合页面”和自动缩放，贴在平整、坚硬、无反光的板上。用尺核对单格 20 mm 和 PDF 底部 100 mm 标尺；横向、纵向都应测量。若方格整体均匀缩放，可将实测边长传给 `--square-mm`；若横纵缩放比例不同，应重新打印。

需要重新生成或改成较小的格子时，在项目根目录执行：

```powershell
python -m pip install -r requirements-calibration.txt
python -m calibration.camera_calibration board --overwrite
# 例如更近距离使用的 10 mm 方格：
python -m calibration.camera_calibration board --square-mm 10 --pdf output/pdf/chessboard_9x6_10mm_A4.pdf --png calibration/boards/chessboard_9x6_10mm_A4.png
```

如果直接使用现成 PDF，采集和标定只依赖项目已有的 OpenCV、NumPy，不需要 reportlab。

## 2. 采集图像

自动调焦相机的正式采集应明确固定焦点，例如：

```powershell
python -m calibration.camera_calibration capture --camera 1 --backend dshow --focus 259 --square-mm 20 --output-dir calibration/focus_259/images
```

259 是此前回位检查使用的驱动读数示例，不保证适合实际定位距离。先在实际工作距离选择清晰的对焦位置；近距离可换较小棋盘，`--square-mm` 必须匹配实物。采集启动后请求关闭自动对焦、设置指定焦点、读帧等待至少 2 秒并核验回读。采集期间在读帧前后检查手动模式和焦点，变化即停止，不混入其他状态。窗口显示 `MANUAL focus=...`。对焦状态来自驱动回读，不是逐帧镜头编码器元数据。

`capture.json` 保存焦点值及逐张状态，标定 JSON 也保存 `focus_position`；不同焦点分别采集和输出，不共用目录。结束时复制控制台打印的实际会话路径，例如：

```powershell
python -m calibration.camera_calibration calibrate --images "calibration/focus_259/images/实际会话目录" --output calibration/focus_259/intrinsics.json
```

一档采集 20–30 张，保持焦点固定，改变棋盘的位置、倾角及清晰范围内的距离。每档至少留出 5–10 张独立验证图放在另一会话，用于验证该档内参。定位程序默认按对焦读数插值或外推内参，标定报告记录各档及插值区间的验证结果。采用单档固定内参定位时，应将相机锁定在对应焦点。

先关闭占用相机的定位程序，固定镜头焦距、对焦和最终使用的分辨率；采集时保持清晰、均匀照明，避免反光和运动模糊。

```powershell
python -m calibration.camera_calibration capture
```

默认读取 `config/positioning_config.json`：当前为 **相机 1、640×480、15 FPS、画面旋转 180°**。窗口中检测成功后显示角点，按 **S 或空格**保存，按 **Q / ESC**退出。保存的是完整、未缩放、无角点标记的 PNG；已经完成与定位相同的旋转，不会在标定时再旋转。

建议拍摄 **20–30 张**，每次改变棋盘的位置、距离、左右倾角及上下倾角。让角点覆盖图像中心、四周和靠近四角的区域；保持整块棋盘及白边可见。不要全部正对相机，也不要连续保存几乎相同的姿态。显示的清晰度是棋盘区域 Laplacian 方差，仅作辅助；默认低于 50 拒绝保存，可通过 `--min-sharpness` 调整。

每次启动创建独立目录，并在控制台打印路径，例如 `calibration/images/session_20261106_153000_123456/`。目录内 `capture.json` 记录分辨率、旋转方向、棋盘规格及相机编号，避免混用。以下覆盖仅对本次采集生效：

```powershell
python -m calibration.camera_calibration capture --camera 0
python -m calibration.camera_calibration capture --square-mm 10
python -m calibration.camera_calibration capture --width 1280 --height 720 --no-flip-180
```

改变分辨率或旋转后，定位程序也必须采用对应设置。格子边长应与打印板实测值一致。

## 3. 标定与检查

将下方路径换成采集结束时打印的会话路径，也可直接复制控制台生成的完整命令：

```powershell
python -m calibration.camera_calibration calibrate --images calibration/images/session_20261106_153000_123456
```

至少需要 10 张检测成功的图像。脚本使用 SB 亚像素棋盘角点检测，失败时回退到传统检测及亚像素优化，再调用 `cv2.calibrateCamera`。

输出：

- `calibration/camera_intrinsics.json`：内参矩阵、畸变系数、整体 RMS、图像尺寸、旋转、棋盘规格、逐张误差、数据来源和质量提示。
- `calibration/camera_intrinsics_diagnostics/corners/`：每张检测成功图片的角点标注，检查是否正确对应。
- `calibration/camera_intrinsics_diagnostics/per_view_errors.csv`：检测状态和逐张重投影 RMSE，单位 px。RMSE 定义为每个角点二维残差平方和的均值再开根号，与 OpenCV 总体 RMS 口径一致。
- `calibration/camera_intrinsics_diagnostics/undistortion_comparison.png`：左原图、右去畸变图。
- `calibration/camera_intrinsics_diagnostics/undistortion_info.json`：仅用于去畸变图的新内参及有效区域。

先检查角点、直线是否变直、边缘覆盖、姿态多样性和逐张误差。低 RMS 不能单独保证定位精度。程序在整体 RMS >1 px、少于 20 张或边缘覆盖不足时给出提示；阈值是排查参考，不是精度认证。

默认保留所有检测成功的视图。如确认某些图像模糊或误检，可移出会话后重算，或明确启用一次误差筛选：

```powershell
python -m calibration.camera_calibration calibrate --images calibration/images/你的会话目录 --max-view-error 1.0 --overwrite
```

逐张 CSV 会保留排除记录；被排除行的误差来自初次拟合，保留行来自最终拟合。重复输出须使用 `--overwrite` 或指定新 `--output`，不会自动改写旧内参。

已有外部照片、没有 `capture.json` 时，必须明确照片方向：

```powershell
# 原始、尚未旋转的照片，要匹配当前定位程序的 180° 旋转：
python -m calibration.camera_calibration calibrate --images D:/my_calibration_photos --cols 9 --rows 6 --square-mm 20 --flip-180
# 若定位本身不旋转，直接使用原始方向：
python -m calibration.camera_calibration calibrate --images D:/my_calibration_photos --cols 9 --rows 6 --square-mm 20 --no-flip-180
```

外部照片应为同一相机、相同分辨率、未裁剪或缩放的图像。不同相机、镜头、对焦、图像尺寸和方向必须分别标定。脚本拒绝混合分辨率，不会自动缩放图像。

## 4. 在定位程序中使用

本项目目前已完成诊断，并将 `config/positioning_config.json` 的 `calibration_file` 指向 `calibration/camera_intrinsics_refined.json`（50 张，RMS 0.542163 px），见 [诊断报告](diagnosis_20261106.md)。以后重新采集并检查结果后，可将 `calibration_file` 改为新的输出路径，例如：

```json
"calibration_file": "../calibration/camera_intrinsics.json"
```

重启 `positioning.py`。此方案保留旧 `calibration/charuco_calib.json`，便于回退。新 JSON 的 `camera_matrix`、`dist_coeffs`、`rms` 可直接加载；定位程序还会核对新文件记录的分辨率、旋转及模型。旧文件没有这些元数据时仍沿用原来的加载方式。

定位程序直接对旋转后的原始图像使用原始 K、D 做 PnP；**不要把去畸变图配原始畸变系数再用于定位**。去畸变对比仅用于检查，去畸变图的新矩阵位于诊断文件。相机内参标定不会重新估计相机安装外参 `T_R_C`。

对其他照片生成对比图：

```powershell
# 本工具采集的图片已经旋转，直接输入：
python -m calibration.camera_calibration undistort --image calibration/images/你的会话目录/frame_0001.png
# 从相机取得、尚未旋转的原始图：
python -m calibration.camera_calibration undistort --image D:/raw_camera_image.png --raw --output calibration/check_raw.png
```

## 验证与参考

```powershell
python -m unittest discover -s tests -p test_camera_calibration.py -v
```

自动测试覆盖已知内参的合成投影恢复、真实角点检测、中文文件路径、旋转元数据、混合分辨率拦截和定位程序兼容性。自动测试不连接实际相机；实际焦距、曝光、打印尺寸及安装精度需在采集时核实。

实现依据：[OpenCV 相机标定教程](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)、[calib3d API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。
