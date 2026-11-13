# 动捕对照、快照回放与标签偏移验证

以下命令从项目根目录运行。原始输入是各批次的 `positioning_records.csv`、动捕 MAT、可用的快照 PNG/JSON，以及该批用到的相机、标签和安装外参。派生结果在各自的 `results/` 目录。重新运行前确认脚本的输出位置。

## 找到正确批次

| 批次目录 | 数据与分析入口 |
|---|---|
| `experiments/26.11.1红外+相机精度测试` | 30 组。主分析用 `analyze_localization.py`，读 `markerPos_test_1.mat` 与 `rigid_body_test_1.csv`，按组建立机翼参考。红外重标定用同目录 `calibrate_infrared.py`。旧 PnP 消融结果在 `results/pnp_replay_20261106/`。 |
| `experiments/26.11.11红外+相机精度测试` | 11.11 第一批。用 `analyze_20261111.py` 和该目录的 `markerPos_test_1.mat`，第 7 组机翼标记建立统一参考。 |
| `experiments/26.11.11红外+相机精度测试2` | 11.11 第二批。用 `analyze_dynamic_focus.py` 和 `markerPos_test_2.mat`，原批次按说明作废第 10 行动捕，评估第 1–9 组。 |
| `experiments/26.11.11红外+相机精度测试3` | 11.11 第三批。动捕为 `markerPos_test_3.mat`。新版布局和逐帧数据用 `replay_snapshots.py`；三维平移试验用 `calibrate_tag_offset.py`。 |

从根目录运行 `python -m experiments.replay_experiment` 可回放 11.1 的 30 组存档。该批次缺少原始快照，补齐后放到该批次的 `camera_snapshots/`，或用 `--snapshots` 指定。复现旧消融还需与历史参与求解标签一致的布局。11.11 第二批的动态内参对照也需要该批次 `camera_snapshots/` 中的逐帧 JSON，当前归档缺少这些文件。

## 动捕参考与误差

1. 复用 [marker_registration.py](../../../../experiments/26.11.1红外+相机精度测试/marker_registration.py)。MAT 的 `markerPos_array` 每个有数据的行按采集组号保留，点坐标从米换到毫米。前三点是机翼标记，后续为机器人标记；机器人点去除重复后应有三个或四个。
2. 标记顺序未知。机翼三点枚举 6 种对应。机器人只有三点时，从四个模型点中选三个并排列，共 24 种。四点齐全时也有 24 种排列。Kabsch 拟合限制为正旋转。按批次的参考构造及物理朝向约束，保留朝向机翼基准平面且残差最低的候选。输出候选对应、残差和采用的参考组。
3. 11.11 三批沿用 [analyze_20261111.py](../../../../experiments/26.11.11红外+相机精度测试/analyze_20261111.py) 的统一机翼参考，第 7 组的机翼标记用于求 `T_W_M`，各组分别用本组机器人标记求动捕机器人位姿。机翼参考与机器人对应关系从动捕和机械约束得到。
4. 视觉位置误差向量为 `p_W_C_vision - p_W_C_gt`，标量为欧氏距离。姿态误差取 `R_W_C_vision @ R_W_C_gt.T` 的相对旋转角。RMSE 是逐组标量误差平方均值的平方根。方向分析按机翼系 XYZ 表达，若转到相机系则使用各组的动捕相机旋转。
5. 红外基准平面是机翼系 `z = -75 mm`。用动捕机器人原点求垂直距离，用动捕姿态把基准法向量与测量法向量转到同一坐标系后求无向夹角。只用状态有效的三路平面数据；第三次测试有效红外平面为 0 组。

## 11.11 第三次测试

```powershell
python "experiments/26.11.11红外+相机精度测试3/replay_snapshots.py"
python "experiments/26.11.11红外+相机精度测试3/calibrate_tag_offset.py"
```

没有图形环境时，回放可加 `--skip-3d`，仍会生成误差表和二维图。两个入口使用本批 `replay_tag_layout.json`，其中 5 号中心为 `(0.3345, 0.0604, -0.075) m`。前者更新 `results/snapshot_replay/`，后者读取该目录并写 `tag_offset_calibration/`。整理目录前生成的回放清单仍记录旧布局路径，拟合偏移前先重新运行回放。

- 10 组动捕对应 11 帧，组 8 有 8A、8B 两帧。现有主方案用先保存的 8A，8B 单独替换用于对照，两帧共享第 8 组动捕参考。8B 对焦读数 496，使用外推内参。MAT 缺少逐组时间戳，两帧与动捕的同步关系需要额外证据。
- `replay_snapshots.py` 按组号与 `snapshot_at` 一一核对 CSV、同名快照 PNG 和 JSON，检查记录的标签 ID、角点、中心、焦距和图像尺寸。直接从带标注 PNG 重检曾在 11 帧中全部漏掉布局内的 5 号标签，主回放使用标注前保存的检测点。
- 第 2 组存在平面 PnP 分支歧义。IPPE 双候选写入 `planar_pose_candidates.csv`；候选用于解释异常，主回放位姿沿用定位程序的解。保留低像素残差与大动捕误差同时出现的记录。
- 默认异常筛选在 8A 主方案的 10 组上分别计算位置、姿态的 `Q3 + 1.5 × (Q3 − Q1)`，一次性剔除任一项超过上界的组。目前剔除 2、4、10。用户若指定只剔除 2 或另有筛选规则，按其口径从全量 `directional_errors.csv` 重算，并说明样本数和理由。重跑默认脚本会重新生成默认筛选结果。

| 回放产物 | 读法 |
|---|---|
| [快照重算误差分析](../../../../experiments/26.11.11红外+相机精度测试3/results/snapshot_replay/快照重算误差分析.md) | 全组和筛选后统计、异常组及可视化 |
| `directional_errors.csv`、`filtered_errors.csv` | 11 帧明细、默认筛选保留组明细 |
| `summary.json`、`direction_summary.json` | 8A 全组汇总、8A/8B 方案及默认筛选统计 |
| `poses.json`、`input_manifest.json` | 主方案视觉和动捕矩阵（毫米）、输入哈希 |

同批次直接位于 `results/` 的旧报告使用原始 CSV 位姿。若要回答新版布局或逐帧回放的问题，查看 `results/snapshot_replay/` 的结果。

## 标签三维平移试验

- 基线用 8A 回放，保持标签旋转、尺寸、内参与安装外参。像点拟合固定动捕相机位姿，最小化四角和中心的加权像素误差；位置拟合取校准组平均机翼系光心误差的反向量。对只使用 5 号标签的帧，给标签中心加 `Δ_W` 等效于给视觉相机光心加同一个 `Δ_W`，相机旋转和视觉 PnP 自身的像素残差在数值误差范围内一致。
- 脚本分别计算全部组、剔除 2、剔除 2/4/10 三种方案。按时间前后近半拆分，交换校准与验证角色，并枚举近半划分检查敏感性。每次偏移只由校准组估计；验证时动捕用于计算误差。合并互换结果时每组只计入其作为验证组的那次。
- 对照验证组的平均位置误差、位置 RMSE 和固定动捕位姿投影的像点 RMSE。剔除名单来自对本批数据的既有分析，保留组指标反映筛选后的表现。不同划分复用同批样本，估计的偏移也可能吸收配准、内参或安装外参误差。将标签物理偏移用于新采集时，以独立数据或实测坐标核验。
- [标签偏移校准验证报告](../../../../experiments/26.11.11红外+相机精度测试3/results/snapshot_replay/tag_offset_calibration/标签偏移校准验证.md) 给出各次估计的 XYZ、验证组及前后指标。同目录有 `validation_errors.csv`、`validation_errors.png` 和含完整划分的 `summary.json`。
