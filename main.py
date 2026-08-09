#!/usr/bin/env python3
"""
AprilTag 实时检测 + 多 Tag 联合单目位姿估计
相机在世界坐标系下的位姿

坐标系定义：
- 机翼模型坐标系：原点在某个固定位置，X轴指向右侧，Y轴指向上侧，Z轴指向前侧，代号W
- 机械臂基本坐标系：原点在机械臂底座中心，X轴指向前方，Y轴指向左侧，Z轴指向上方，代号armbase
- 机械臂末端坐标系：原点在机械臂末端中心，X轴指向下侧，Y轴指向左边侧，Z轴指向前侧，代号armend
- 相机坐标系：原点在相机光心，X轴指向右侧，Y轴指向下侧，Z轴指向相机前方，代号cam
- Tag坐标系：原点在Tag中心，X轴指向Tag的右侧，Y轴指向下侧，Z轴垂直纸面向里

T_A_B 表示从坐标系 A 到坐标系 B 的变换矩阵，满足 p_B = T_A_B * p_A
在output中的TAD.npy表示从机械臂末端坐标系到世界坐标系的变换矩阵

本程序中单位均采用米和角度

control shift P 启动

需要:
  pip install opencv-python pupil-apriltags numpy
"""

import cv2
import numpy as np
from pupil_apriltags import Detector
import rclpy
import os
import sys
import threading
from rclpy.executors import MultiThreadedExecutor
import time

# 如果 nodes 报错，请取消下面两行的注释并确保路径正确
piper_path = "/home/gavin/piper_ws/src/piper-beginner-guide/piper_ros_control/piper_ros_control"
sys.path.insert(0, piper_path)

from nodes.piper_controller import PiperController

# ====================== 用户配置 ======================
FAMILY = "tag36h11"          # 与生成时一致
TAG_SIZE = 0.030              # 真实物理边长（米），必须精确测量！
CAMERA_INDEX = 2             # 摄像头编号

T_armend_cam = np.array([[0, 1, 0, 0.0],   # 相机在机械臂末端的位姿（末端坐标系下）
                        [-1, 0, 0, 0.018],
                        [0, 0, 1, 0.056],
                        [0, 0, 0, 1]], dtype=np.float64)

# ---------- 世界坐标系中每个 Tag 的已知位姿 ----------
# 格式: tag_id → {"tvec": [x, y, z], "rvec": [rx, ry, rz]}  (单位: 米 / 弧度)
# rvec 是 Rodrigues 旋转向量，也可用下面的 helper 从欧拉角生成
#
# 坐标系约定（与 AprilTag 官方一致）：
#   Tag 局部坐标系: 中心原点，X 右，Y 下，Z 朝向相机（出 tag 平面）
#   世界坐标系由你自定义（例如地面为 XY，Z 向上）

def rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=0):
    """简单 ZYX 欧拉角 (度) → Rodrigues rvec。方便定义 tag 朝向。"""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    # ZYX 欧拉角组合：先绕 X，再绕 Y，最后绕 Z
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()

def pose_to_matrix(x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg):
    """将机械臂位姿（米、度）转换为齐次变换矩阵。"""
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    # 绕 X 轴旋转 roll
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    # 绕 Y 轴旋转 pitch
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0,              1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    # 绕 Z 轴旋转 yaw
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    # ZYX 欧拉角组合：先绕 X，再绕 Y，最后绕 Z
    R_mat = Rz @ Ry @ Rx

    # 组成 4x4 齐次矩阵
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [x_m, y_m, z_m]
    return T

# ========== 在这里配置你的 tags 世界位姿（示例，请按实际修改！） ==========
WORLD_TAG_POSES = {
    # 示例1: ID=0，放在世界原点，tag 平面与世界 XY 平行，Z 朝上（如果世界 Z 上）
    # 注意：若 tag 贴在墙上/地板，请调整 rvec

    #----------------0 1 2为左边三个--------------------
    0: {
        "tvec": np.array([0.0, 0.065, 0.055], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=90, roll_deg=180)   # 或直接 np.zeros(3)
    },
    1: {
        "tvec": np.array([0.0, 0.095, -0.015], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=90, roll_deg=180)
    },
    2: {
        "tvec": np.array([0.0, 0.045, -0.015], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=90, roll_deg=180)
    },

    # #----------------3至12为正前方--------------------
    3: {
        "tvec": np.array([0.065, 0.070, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },
    4: {
        "tvec": np.array([0.135, 0.070, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },
    5: {
        "tvec": np.array([0.205, 0.070, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },
    6: {
        "tvec": np.array([0.275, 0.100, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },
    7: {
        "tvec": np.array([0.345, 0.100, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },
    8: {
        "tvec": np.array([0.275, 0.050, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },
    9: {
        "tvec": np.array([0.345, 0.050, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },
    10: {
        "tvec": np.array([0.415, 0.070, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },
    11: {
        "tvec": np.array([0.485, 0.070, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },
    12: {
        "tvec": np.array([0.600, 0.070, -0.05], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=180)
    },

    #--------------13至15为上方三个------------------------
    13: {
        "tvec": np.array([0.32678, 0.15167, 0.060], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=-90)
    },
    14: {
        "tvec": np.array([0.29678, 0.15167, -0.005], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=-90)
    },
    15: {
        "tvec": np.array([0.35892, 0.15167, -0.005], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=0, pitch_deg=0, roll_deg=-90)
    },

    #----------------16至18为左下三个------------------------
    16: {
        "tvec": np.array([0.18331, 0.01403, -0.005], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=-1.3, pitch_deg=0, roll_deg=90)
    },
    17: {
        "tvec": np.array([0.24544, 0.01262, -0.005], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=-1.3, pitch_deg=0, roll_deg=90)
    },
    18: {
        "tvec": np.array([0.21331, 0.01335, 0.060], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=-1.3, pitch_deg=0, roll_deg=90)
    },

    #----------------19至21为右下三个------------------------
    19: {
        "tvec": np.array([0.45456, 0.01262, -0.005], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=1.3, pitch_deg=0, roll_deg=90)
    },
    20: {
        "tvec": np.array([0.51669, 0.01403, -0.005], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=1.3, pitch_deg=0, roll_deg=90)
    },
    21: {
        "tvec": np.array([0.48456, 0.01330, 0.060], dtype=np.float64),
        "rvec": rvec_from_euler(yaw_deg=1.3, pitch_deg=0, roll_deg=90)
    }
}

# 如果你更喜欢用 4x4 齐次变换矩阵，也可以改成:
# WORLD_TAG_POSES = {0: T_world_tag0_4x4, ...} 然后自行调整 get_tag_corners_world
# =================================================================

try:
    with np.load("charuco_calib.npz") as data:  # 修复：使用 with 自动关闭
        print("成功加载相机参数")
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
except Exception:
    print("未成功加载相机参数，采用默认值（精度会差，请标定！）")
    fx = fy = 600.0
    cx, cy = 320.0, 240.0
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

# ====================================================

def get_tag_corners_local(tag_size):
    """Tag 局部坐标系下的 4 个角点（与 pupil-apriltags + 原代码顺序一致）
    顺序: BL → BR → TR → TL  (CCW，Y 向下)
    注意：请用实拍验证 tag.corners 是否严格按此顺序对应！
    """
    s = tag_size / 2.0
    return np.array([
        [-s,  s, 0],   # 0: bottom-left
        [ s,  s, 0],   # 1: bottom-right
        [ s, -s, 0],   # 2: top-right
        [-s, -s, 0]    # 3: top-left
    ], dtype=np.float64)

def get_tag_corners_world(tag_id, tag_size, world_poses):
    """根据已知世界位姿，计算 tag 4 个角点在世界坐标系下的坐标"""
    if tag_id not in world_poses:
        return None
    pose = world_poses[tag_id]
    tvec = np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1)
    rvec = np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    local_corners = get_tag_corners_local(tag_size)          # (4,3)
    world_corners = (R @ local_corners.T + tvec).T           # (4,3)
    return world_corners

def draw_axis(img, rvec, tvec, K, D, length=0.05):
    """画坐标轴"""
    try:
        cv2.drawFrameAxes(img, K, D, rvec, tvec, length)
    except Exception:
        axis_pts = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]])
        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, D)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        origin = tuple(imgpts[0])
        cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X 红
        cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y 绿
        cv2.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z 蓝

def pose_inverse(rvec, tvec):
    """从 T_obj_cam 求 T_cam_obj （相机在物体/世界坐标系下的位姿）"""
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec.reshape(3, 1)
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv, t_inv

def rotation_to_euler_zyx_deg(R):
    """将 R = Rz(yaw) @ Ry(pitch) @ Rx(roll) 转为度。"""
    sy = np.clip(-R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # 万向节锁时固定 yaw=0，保留一个稳定表示
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.rad2deg([roll, pitch, yaw])

def main():
    rclpy.init()
    arm = PiperController("piper_pose_reader")

    executor = MultiThreadedExecutor()
    executor.add_node(arm)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    T_W_armbase = np.linalg.inv(np.load("output/TAD.npy"))
    T_W_armbase[:3, 3:] = T_W_armbase[:3, 3:] / 1000.0  # mm → m

    cap = None  # 用于 finally 清理
    try:
        print("[INFO] Piper ROS 控制器已启动，等待机械臂位姿反馈...")
        timeout = 5.0  # 建议稍长一点
        start_time = time.time()
        enabled = False
        while (time.time() - start_time < timeout):
            if getattr(arm, "_is_enabled", False):
                print("[INFO] Piper 已使能")
                print("当前位姿:", getattr(arm, "_current_pose", None))
                enabled = True
                break
            print("[WARN] Piper 未使能或未连接，等待中...")
            time.sleep(0.2)

        if not enabled:
            print("[ERROR] PiperController 未收到机械臂位姿")
            print("[ERROR] 请检查 ROS 节点、话题、机械臂驱动是否正常启动")
            return  # finally 会清理

        detector = Detector(
            families=FAMILY,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        local_obj_pts = get_tag_corners_local(TAG_SIZE)   # 预计算，单 tag 时用

        # 使用 V4L2 + MJPEG，降低高分辨率采集时的 USB 带宽占用
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("无法打开摄像头")
            return  # finally 会清理

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_text = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
        print(f"摄像头实际参数: {actual_width}x{actual_height}, "
              f"fps={actual_fps:.1f}, fourcc={fourcc_text}")
        if (actual_width, actual_height) != (640, 480):
            print("[ERROR] 摄像头未运行在 640x480；当前标定参数不匹配，请重新标定或调整摄像头模式")
            return  # finally 会清理

        print("按 q 退出，空格暂停，s 保存/打印当前位姿对比")
        print(f"已配置 {len(WORLD_TAG_POSES)} 个 tag 的世界位姿")
        print("检测到多个已知 tag 时将进行联合 PnP，精度更好\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 无法读取摄像头帧")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray,
                                   estimate_tag_pose=False,
                                   camera_params=None,
                                   tag_size=None)

            # ---------- 收集用于联合 PnP 的点 ----------
            all_world_pts = []
            all_img_pts = []
            valid_tags = []          # 用于后续单 tag 可视化
            tag_inlier_counts = None # 联合 PnP 成功后，每个 tag 的角点内点数
            inlier_point_indices = None  # 全部世界点中属于 RANSAC 内点的索引

            # 每帧重置
            camera_pose_valid = False
            R_cam = None
            t_cam = None
            rvec = None
            tvec = None

            # 读取机械臂反馈，实时计算相机真实位姿（机械臂基坐标系）
            real_pose_text = "Arm camera: waiting for pose..."
            current_pose = getattr(arm, "_current_pose", None)
            if current_pose is not None:
                try:
                    T_armbase_armend = pose_to_matrix(
                        current_pose.x, current_pose.y, current_pose.z,
                        current_pose.roll, current_pose.pitch, current_pose.yaw
                    )
                    T_W_cam_real = T_W_armbase @ T_armbase_armend @ T_armend_cam
                    real_xyz = T_W_cam_real[:3, 3]
                    real_rpy = rotation_to_euler_zyx_deg(T_W_cam_real[:3, :3])
                    real_pose_text = (
                        f"Arm camera: x={real_xyz[0]:.3f} "
                        f"y={real_xyz[1]:.3f} z={real_xyz[2]:.3f} m | "
                        f"\n"
                        f"r={real_rpy[0]:.1f} p={real_rpy[1]:.1f} "
                        f"y={real_rpy[2]:.1f} deg"
                    )
                except Exception as e:
                    real_pose_text = f"Arm camera: pose error ({e})"

            for tag in tags:
                # 画框 + ID（始终做）
                corners = tag.corners.astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                c = tag.center.astype(int)
                cv2.circle(frame, tuple(c), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{tag.tag_id}",
                            (c[0] - 20, c[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 检查是否有世界位姿
                world_corners = get_tag_corners_world(tag.tag_id, TAG_SIZE, WORLD_TAG_POSES)
                if world_corners is not None:
                    all_world_pts.append(world_corners)
                    all_img_pts.append(tag.corners.astype(np.float64))
                    valid_tags.append(tag)

                    # 也能算单 tag 到相机的位姿，用于画该 tag 的坐标轴
                    img_pts = tag.corners.astype(np.float64).reshape(-1, 1, 2)
                    success, rvec_tag, tvec_tag = cv2.solvePnP(
                        local_obj_pts, img_pts,
                        camera_matrix, dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if success:
                        draw_axis(frame, rvec_tag, tvec_tag, camera_matrix, dist_coeffs, length=TAG_SIZE * 0.6)

            # ---------- 多 Tag 联合求解相机位姿（世界系） ----------
            if len(all_world_pts) >= 1:          # 至少 1 个 tag（4 点），多 tag 更好
                world_pts = np.vstack(all_world_pts).astype(np.float64)   # (N*4, 3)
                img_pts   = np.vstack(all_img_pts).astype(np.float64).reshape(-1, 1, 2)

                # 修复/增强：使用 RANSAC 提高鲁棒性，过滤 outlier
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    world_pts, img_pts,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,  # 或 SOLVEPNP_SQPNP
                    reprojectionError=100.0,
                    confidence=0.99,
                    iterationsCount=100
                )

                if success and inliers is not None and len(inliers) >= 4:
                    camera_pose_valid = True
                    inlier_point_indices = set(inliers.reshape(-1).tolist())
                    tag_inlier_counts = {
                        tag.tag_id: sum(
                            point_idx in inlier_point_indices
                            for point_idx in range(tag_idx * 4, tag_idx * 4 + 4)
                        )
                        for tag_idx, tag in enumerate(valid_tags)
                    }
                    # rvec, tvec 是 world → camera 的变换
                    # 我们需要 camera 在 world 下的位姿
                    rvec_cam, tvec_cam = pose_inverse(rvec, tvec)

                    t_cam = tvec_cam.flatten()
                    R_cam, _ = cv2.Rodrigues(rvec_cam)
                    roll_cam, pitch_cam, yaw_cam = rotation_to_euler_zyx_deg(R_cam)
                    orth_error = np.linalg.norm(R_cam.T @ R_cam - np.eye(3))
                    det_R = np.linalg.det(R_cam)

                    # 在画面上显示结果
                    num_inliers = len(inliers)
                    info = f"Camera @ World: x={t_cam[0]:.3f} y={t_cam[1]:.3f} z={t_cam[2]:.3f} m \n ({len(valid_tags)} tags, {num_inliers} inliers)"
                    cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 控制台输出（改用普通 print，避免 \r 问题）
                    print(f"[多Tag] 使用 {len(valid_tags)} 个 tag | inliers={num_inliers} | "
                          f"相机世界位置: [{t_cam[0]:7.3f}, {t_cam[1]:7.3f}, {t_cam[2]:7.3f}] m")
                    print(f"[6D姿态] xyz(m)=({t_cam[0]:.4f}, {t_cam[1]:.4f}, {t_cam[2]:.4f}) | "
                          f"rpy(deg)=({roll_cam:.2f}, {pitch_cam:.2f}, {yaw_cam:.2f}) | "
                          f"R正交误差={orth_error:.2e}, det(R)={det_R:.6f}")

                    # 绘制世界坐标轴：由相机在世界系下的位姿重新构造
                    # world -> camera，避免直接复用 RANSAC rvec 的表示分支。
                    R_world_to_cam = R_cam.T
                    t_world_to_cam = -R_world_to_cam @ t_cam.reshape(3, 1)
                    rvec_world_to_cam, _ = cv2.Rodrigues(R_world_to_cam)
                    draw_axis(frame, rvec_world_to_cam, t_world_to_cam,
                              camera_matrix, dist_coeffs, length=0.05)
                else:
                    cv2.putText(frame, "PnP failed or too few inliers", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No known tags detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 实时显示机械臂+手眼变换得到的相机真实位姿
            cv2.putText(frame, real_pose_text, (40, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)

            # 在每个参与联合 PnP 的 tag 下方显示其 4 个角点中的内点数
            for tag_idx, tag in enumerate(valid_tags):
                center = tag.center.astype(int)
                if tag_inlier_counts is None:
                    inlier_text = "inliers: -/4"
                    color = (0, 165, 255)
                else:
                    count = tag_inlier_counts.get(tag.tag_id, 0)
                    inlier_text = f"inliers: {count}/4"
                    color = (0, 255, 0) if count == 4 else (0, 0, 255)
                cv2.putText(frame, inlier_text,
                            (center[0] - 35, center[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 标记每个角点是否为 RANSAC 内点
                corners = tag.corners.astype(int)
                for corner_idx, (x, y) in enumerate(corners):
                    point_idx = tag_idx * 4 + corner_idx
                    if inlier_point_indices is None:
                        point_text = f"P{corner_idx}:?"
                        point_color = (0, 165, 255)
                    elif point_idx in inlier_point_indices:
                        point_text = f"P{corner_idx}:OK"
                        point_color = (0, 255, 0)
                    else:
                        point_text = f"P{corner_idx}:OUT"
                        point_color = (0, 0, 255)
                    cv2.circle(frame, (x, y), 6, point_color, -1)
                    cv2.putText(frame, point_text, (x + 7, y - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, point_color, 2)

            cv2.imshow("AprilTag Multi-Tag Pose (World)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    except Exception as e:
        print(f"[ERROR] 主循环异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ========== 关键：确保资源总是被清理 ==========
        print("\n[INFO] 正在清理资源...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        try:
            arm.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("已退出")

if __name__ == "__main__":
    main()
