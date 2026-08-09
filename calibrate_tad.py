#!/usr/bin/env python3
"""
机翼模型 -> 机械臂基坐标系 (TAD) 标定

工作流：
  1. 启动后机械臂进入示教模式 (扭矩关闭, 可手动拖动)
  2. 程序内置 3 个标定点在机翼模型坐标系 (D) 下的坐标 (mm),
     用户手动移动机械臂, 使夹具边角依次对齐这 3 个点
  3. 用 Kabsch (SVD) 算法求 TAD (机翼模型 D -> 机械臂基 A)
  4. 保存 TAD 到 output/TAD.npy (并附时间戳备份), 供 main.py 加载

约定：
  TAB 把 B 坐标系下的点变换到 A 坐标系下 (即 T_A_B), p_A = T_A_B @ p_B
  A: 机械臂基坐标系
  B: 机械臂末端坐标系
  D: 机翼模型坐标系

单位约定（重要）：
  arm._current_pose 反馈是 (m, deg).
  程序内部统一使用 (mm, deg). 在读取 pose 时做一次单位转换.
  fixture_corner_B, points_D 全部使用 mm.
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor

import time
import threading
import numpy as np
import os
import argparse

from scipy.spatial.transform import Rotation as Rot

from piper_ros_control.nodes.piper_controller import PiperController


class TADCalibrator:
    def __init__(
        self,
        save_dir="output",
        save_path="output/TAD.npy",
        fixture_corner_B=(0.0, 18.0, 56.0),
    ):
        """
        参数：
            save_dir:
                标定结果保存目录 (TAD.npy + 带时间戳的备份 + txt 记录).

            save_path:
                TAD 的默认输出文件 (main.py 默认从这里加载).

            fixture_corner_B:
                夹具边角在机械臂末端 (frame B) 坐标系下的位置, 单位 mm.
                示教时用这个点对准机翼模型上的 3 个已知标定点.
        """

        self.arm = PiperController("piper_tad_calibrator")

        self.save_dir = save_dir
        self.save_path = save_path
        os.makedirs(self.save_dir, exist_ok=True)

        self.fixture_corner_B = np.array(fixture_corner_B, dtype=float).reshape(-1)
        if self.fixture_corner_B.shape != (3,):
            raise ValueError(
                f"fixture_corner_B 必须是 3 元素 (x, y, z), 当前 shape={self.fixture_corner_B.shape}"
            )

    # ============================================================
    # (mm, deg)  <->  4x4 齐次变换
    # ============================================================
    @staticmethod
    def pose_to_T(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg):
        """(x, y, z, roll, pitch, yaw) [mm, deg]  ->  4x4 齐次变换矩阵.

        旋转约定: scipy 'xyz' (extrinsic) = Rz(yaw) Ry(pitch) Rx(roll).
        与 Piper 反馈的旋转约定一致.
        """
        T = np.eye(4, dtype=float)
        T[:3, :3] = Rot.from_euler(
            "xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True
        ).as_matrix()
        T[:3, 3] = [float(x_mm), float(y_mm), float(z_mm)]
        return T

    # ============================================================
    # 从实机反馈构建 TAB (单位转换: m -> mm)
    # ============================================================
    def build_TAB_from_current_pose(self):
        """读取 arm._current_pose (单位 m + rad) 并构建 TAB (内部 mm + deg)."""
        pose = self.arm._current_pose
        print(f"[DEBUG] arm._current_pose = {pose}")
        if pose is None:
            raise RuntimeError("无法读取机械臂当前位姿 (arm._current_pose is None)")

        return self.pose_to_T(
            pose.x * 1000.0, pose.y * 1000.0, pose.z * 1000.0,
            pose.roll, pose.pitch, pose.yaw
        )

    # ============================================================
    # SVD (Kabsch) 求最优刚体变换
    # ============================================================
    @staticmethod
    def solve_rigid_transform(points_src, points_dst):
        """求 4x4 齐次 T 使得 [dst; 1] = T @ [src; 1] 最小二乘最优."""
        src = np.asarray(points_src, dtype=float)
        dst = np.asarray(points_dst, dtype=float)

        if src.shape != dst.shape or src.shape[1] != 3 or src.shape[0] < 3:
            raise ValueError(
                f"输入点集形状错误: src={src.shape}, dst={dst.shape}, 需要 (N>=3, 3)"
            )

        centroid_src = src.mean(axis=0)
        centroid_dst = dst.mean(axis=0)

        src_c = src - centroid_src
        dst_c = dst - centroid_dst

        H = src_c.T @ dst_c
        U, _, Vt = np.linalg.svd(H)

        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])

        R = Vt.T @ D @ U.T
        t = centroid_dst - R @ centroid_src

        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    # ============================================================
    # 主流程: 3 点对齐求 TAD
    # ============================================================
    def calibrate(self, points_D):
        """
        参数:
            points_D: (3, 3) 数组, 3 个标定点在机翼模型坐标系下的坐标 (mm).
        """
        points_D = np.asarray(points_D, dtype=float)
        if points_D.shape != (3, 3):
            raise ValueError(f"points_D 必须形状 (3, 3), 当前 {points_D.shape}")

        print("\n" + "=" * 60)
        print("机翼模型 -> 机械臂基 (TAD) 标定 (3 点对齐)")
        print("=" * 60)
        print(f"夹具边角 (B 坐标系, mm): {self.fixture_corner_B}")
        print(f"\n[CALIB] D 坐标系下的标定点 (mm):\n{points_D}")

        v1 = points_D[1] - points_D[0]
        v2 = points_D[2] - points_D[0]
        if np.linalg.norm(np.cross(v1, v2)) < 1e-3:
            print("[WARN] 3 个标定点几乎共线, 标定姿态会不可靠! 建议重选.")

        # 切到示教模式
        print("\n[TEACH] 进入示教模式 (机械臂关闭扭矩, 可手动拖动)")
        try:
            print("[TEACH] arm.disable() 已调用")
        except AttributeError:
            print("[WARN] PiperController 没有 disable() 方法, "
                  "请用物理急停或外部脚本手动切到示教模式")
        except Exception as e:
            print(f"[WARN] arm.disable() 调用失败: {e}")
        time.sleep(1.0)

        corner_B_h = np.array([
            self.fixture_corner_B[0],
            self.fixture_corner_B[1],
            self.fixture_corner_B[2],
            1.0,
        ], dtype=float)

        points_A = []
        i = 0
        while i < 3:
            print(f"\n[ALIGN] 请手动移动机械臂, 让夹具边角对准机翼上的标定点 {i + 1}")
            print(f"        该点在机翼模型坐标系下: {points_D[i]} mm")
            input(f"        对齐完成后按 Enter 记录...")

            try:
                TAB_i = self.build_TAB_from_current_pose()
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                print("[RETRY] 请确认机械臂反馈正常后重新对齐")
                continue

            corner_A = TAB_i @ corner_B_h
            points_A.append(corner_A[:3])

            print(f"[REC]   实机末端位姿 (mm): {TAB_i[:3, 3]}")
            print(f"[REC]   夹具边角在 A 坐标系: {corner_A[:3]} mm")
            i += 1

        points_A = np.array(points_A, dtype=float)

        TAD = self.solve_rigid_transform(points_D, points_A)

        pts_pred = (TAD @ np.hstack([points_D, np.ones((3, 1))]).T).T[:, :3]
        residuals = np.linalg.norm(pts_pred - points_A, axis=1)

        print("\n" + "-" * 60)
        print("[CALIB] 标定完成")
        print(f"TAD =\n{TAD}")
        print(f"每点残差 (mm): {residuals}")
        print(f"平均残差 (mm): {residuals.mean():.4f}")
        print("-" * 60)

        self.save_result(TAD, points_D, points_A, residuals)

        return TAD

    # ============================================================
    # 保存标定结果
    # ============================================================
    def save_result(self, TAD, points_D, points_A, residuals):
        ts = int(time.time())
        backup_npy = os.path.join(self.save_dir, f"TAD_calibration_{ts}.npy")
        txt_path = os.path.join(self.save_dir, f"TAD_calibration_{ts}.txt")

        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        np.save(self.save_path, TAD)
        np.save(backup_npy, TAD)

        with open(txt_path, "w") as f:
            f.write("# Wing model -> Arm base calibration\n")
            f.write(f"timestamp: {ts}\n\n")
            f.write(f"fixture_corner_B (mm): {self.fixture_corner_B.tolist()}\n\n")
            f.write("points_D (mm, wing model frame):\n")
            for p in points_D:
                f.write(f"  {p.tolist()}\n")
            f.write("\npoints_A (mm, arm base frame):\n")
            for p in points_A:
                f.write(f"  {p.tolist()}\n")
            f.write("\nTAD (4x4):\n")
            for row in TAD:
                f.write("  " + " ".join(f"{v:.6f}" for v in row) + "\n")
            f.write(f"\nresiduals (mm): {residuals.tolist()}\n")
            f.write(f"mean residual (mm): {residuals.mean():.6f}\n")

        print(f"[SAVE] 主输出: {self.save_path}  (main.py 默认读取)")
        print(f"[SAVE] 备份:   {backup_npy}")
        print(f"[SAVE] 记录:   {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wing model -> Arm base calibration (3-point alignment)"
    )
    parser.add_argument("--save_dir", type=str, default="output",
                        help="标定结果保存目录")
    parser.add_argument("--save_path", type=str, default="output/TAD.npy",
                        help="TAD 主输出文件 (main.py 默认读取)")
    parser.add_argument("--fixture_corner", type=float, nargs=3,
                        default=[0.0, 18.0, 56.0], metavar=("X", "Y", "Z"),
                        help="夹具边角在末端坐标系下的位置 [mm]")
    parser.add_argument("--points_D", type=float, nargs=9,
                        default=[155.56, 14.66, 100.00,
                                 426.81, 11.99, 100.00,
                                 266.78, 170.44, 100.00],
                        metavar="V",
                        help="3 个标定点在机翼坐标系下的坐标 (mm), "
                             "9 个数: x1 y1 z1 x2 y2 z2 x3 y3 z3")
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()

    cal = None
    try:
        cal = TADCalibrator(
            save_dir=args.save_dir,
            save_path=args.save_path,
            fixture_corner_B=tuple(args.fixture_corner),
        )

        executor = MultiThreadedExecutor()
        executor.add_node(cal.arm)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        print("[INIT] Waiting for ROS2 nodes to connect...")
        time.sleep(2.0)

        points_D = np.array(args.points_D, dtype=float).reshape(3, 3)
        cal.calibrate(points_D)

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")

    finally:
        print("[CLEANUP] Cleaning up...")
        if cal is not None:
            try:
                cal.arm.destroy_node()
            except Exception as e:
                print(f"[WARN] Failed to destroy arm node: {e}")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
