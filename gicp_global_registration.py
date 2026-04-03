import os
import copy
import time
from typing import Optional, Tuple, List

import numpy as np
import open3d as o3d
import pyrealsense2 as rs


class GlobalGICPVerifier:
    """
    单纯使用 GICP 做“当前帧 -> 全局地图”的验证脚本。

    改进点：
    1. 可视化更顺滑，主循环里始终先刷新 UI。
    2. 重计算期间减少几何体重建频率。
    3. 降低默认搜索密度，减少窗口卡顿。
    """

    def __init__(self):
        # ---------- 运行参数 ----------
        self.map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_environment_map_pgo.ply")
        self.process_every_n_frames = 8
        self.depth_min = 0.07
        self.depth_max = 0.6

        # ---------- GICP 参数 ----------
        self.local_voxel = 0.003
        self.global_voxel = 0.025
        self.global_coarse_threshold = 0.06
        self.global_fine_threshold = 0.025
        self.final_threshold = 0.01

        # 搜索空间（纯 GICP 全局搜索）
        self.yaw_candidates = np.arange(-180, 180, 45)
        self.xy_search_range = 0.35
        self.xy_step = 0.18
        self.top_k = 3

        # 结果接受阈值
        self.accept_fitness = 0.35
        self.accept_rmse = 0.03

        # ---------- Realsense ----------
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)

        self.intrinsics = None
        self.depth_scale = 0.001
        self.depth_intrinsics = None

        # ---------- 数据 ----------
        self.map_pcd: Optional[o3d.geometry.PointCloud] = None
        self.map_global_down: Optional[o3d.geometry.PointCloud] = None
        self.frame_counter = 0
        self.last_result = None

        # ---------- 可视化 ----------
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.current_frame_vis: Optional[o3d.geometry.PointCloud] = None
        self.vis_frame_skip = 2
        self._vis_loop_counter = 0
        self._last_ui_refresh = 0.0
        self.ui_refresh_interval_sec = 0.01
        self.keep_running = True

    # -----------------------------
    # 基础功能
    # -----------------------------
    def start_stream(self) -> bool:
        try:
            profile = self.pipeline.start(self.config)
            print("Realsense stream started.")

            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intrinsics = color_stream.get_intrinsics()
            self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

            print(
                f"Intrinsics fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
                f"cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}"
            )
            print(f"Depth scale = {self.depth_scale}")
            return True
        except Exception as e:
            print(f"Failed to start Realsense stream: {e}")
            return False

    def stop_stream(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def load_map(self, map_path: str) -> bool:
        try:
            if not os.path.exists(map_path):
                print(f"地图不存在: {map_path}")
                return False

            self.map_pcd = o3d.io.read_point_cloud(map_path)
            if self.map_pcd is None or len(self.map_pcd.points) == 0:
                print("地图点云为空")
                return False

            self.map_pcd = self.preprocess_local_pcd(self.map_pcd)
            self.map_global_down = self.preprocess_global_pcd(self.map_pcd)
            print(f"Map loaded: {map_path}")
            print(f"Map points(local): {len(self.map_pcd.points)}")
            print(f"Map points(global): {len(self.map_global_down.points)}")
            return True
        except Exception as e:
            print(f"加载地图失败: {e}")
            return False

    def setup_visualizer(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Global GICP Verify", width=1280, height=720)
        self.vis.add_geometry(self.map_pcd)
        self.vis.add_geometry(self.pose_frame)
        self.current_frame_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.current_frame_vis)

        def _quit_callback(_vis):
            self.keep_running = False
            return False

        self.vis.register_key_callback(ord("Q"), _quit_callback)
        self.vis.register_key_callback(256, _quit_callback)  # ESC

        opt = self.vis.get_render_option()
        opt.point_size = 2.0

    def refresh_ui(self, force: bool = False):
        if self.vis is None:
            return
        now = time.time()
        if not force and (now - self._last_ui_refresh) < self.ui_refresh_interval_sec:
            return
        self.vis.poll_events()
        self.vis.update_renderer()
        self._last_ui_refresh = now

    def capture_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        return depth_frame, color_frame

    # -----------------------------
    # 点云预处理
    # -----------------------------
    def depth_to_pointcloud(self, depth_frame, color_frame) -> o3d.geometry.PointCloud:
        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        h, w = depth.shape
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy

        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32) * self.depth_scale

        mask = (z > self.depth_min) & (z < self.depth_max)
        if np.count_nonzero(mask) == 0:
            return o3d.geometry.PointCloud()

        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy

        pts = np.stack([x[mask], y[mask], z[mask]], axis=1).astype(np.float64)
        cols = color[mask][:, ::-1].astype(np.float64) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd

    def preprocess_local_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        down = pcd.voxel_down_sample(self.local_voxel)
        if len(down.points) > 30:
            down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.local_voxel * 2.0,
                    max_nn=30,
                )
            )
        return down

    def preprocess_global_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        down = pcd.voxel_down_sample(self.global_voxel)
        if len(down.points) > 30:
            down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.global_voxel * 2.0,
                    max_nn=30,
                )
            )
        return down

    # -----------------------------
    # GICP 全局搜索
    # -----------------------------
    def get_translation_candidates(self) -> List[Tuple[float, float]]:
        r = self.xy_search_range
        s = self.xy_step
        xs = np.arange(-r, r + 1e-6, s)
        ys = np.arange(-r, r + 1e-6, s)
        return [(float(x), float(y)) for x in xs for y in ys]

    @staticmethod
    def make_xyyaw_transform(x: float, y: float, yaw_deg: float) -> np.ndarray:
        yaw = np.deg2rad(yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:3, :3] = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        T[0, 3] = x
        T[1, 3] = y
        return T

    @staticmethod
    def score_result(result) -> float:
        return float(result.fitness - 3.0 * result.inlier_rmse)

    def global_gicp_register(self, source_raw: o3d.geometry.PointCloud):
        source_global = self.preprocess_global_pcd(source_raw)
        target_global = self.map_global_down

        if len(source_global.points) < 50:
            print("当前帧有效点太少，跳过")
            return None

        coarse_pool = []
        translations = self.get_translation_candidates()

        total_candidates = len(translations) * len(self.yaw_candidates)
        checked = 0

        for x, y in translations:
            for yaw in self.yaw_candidates:
                checked += 1
                if checked % 5 == 0:
                    self.refresh_ui()
                init_T = self.make_xyyaw_transform(x, y, yaw)
                try:
                    result = o3d.pipelines.registration.registration_generalized_icp(
                        source_global,
                        target_global,
                        self.global_coarse_threshold,
                        init_T,
                        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            relative_fitness=1e-5,
                            relative_rmse=1e-5,
                            max_iteration=18,
                        ),
                    )
                except Exception:
                    continue

                if result.fitness < 0.05:
                    continue

                score = self.score_result(result)
                coarse_pool.append((score, result, x, y, yaw))

        if not coarse_pool:
            return None

        coarse_pool.sort(key=lambda item: item[0], reverse=True)
        coarse_pool = coarse_pool[: self.top_k]

        best = None
        best_score = -1e9

        for _, coarse_res, x, y, yaw in coarse_pool:
            self.refresh_ui()
            try:
                fine_res = o3d.pipelines.registration.registration_generalized_icp(
                    source_global,
                    target_global,
                    self.global_fine_threshold,
                    coarse_res.transformation,
                    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=30,
                    ),
                )
            except Exception:
                continue

            source_local = self.preprocess_local_pcd(source_raw)
            final_res = o3d.pipelines.registration.registration_generalized_icp(
                source_local,
                self.map_pcd,
                self.final_threshold,
                fine_res.transformation,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-7,
                    relative_rmse=1e-7,
                    max_iteration=35,
                ),
            )

            score = self.score_result(final_res)
            t = final_res.transformation[:3, 3]
            print(
                f"[Global GICP] x0={x:+.2f}, y0={y:+.2f}, yaw0={yaw:+4.0f} | "
                f"fitness={final_res.fitness:.3f}, rmse={final_res.inlier_rmse:.4f}, "
                f"t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}), score={score:.3f}"
            )

            if score > best_score:
                best_score = score
                best = final_res

        return best

    # -----------------------------
    # 可视化
    # -----------------------------
    def update_visualization(self, current_pcd: o3d.geometry.PointCloud, transformation: np.ndarray):
        if self.vis is None:
            return

        self._vis_loop_counter += 1
        if self._vis_loop_counter % self.vis_frame_skip != 0:
            self.refresh_ui()
            return

        current_vis = copy.deepcopy(current_pcd)
        current_vis.transform(transformation)

        self.current_frame_vis.points = current_vis.points
        if current_vis.has_colors():
            self.current_frame_vis.colors = current_vis.colors

        pose_frame_copy = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        pose_frame_copy.transform(transformation)
        self.pose_frame.vertices = pose_frame_copy.vertices
        self.pose_frame.triangles = pose_frame_copy.triangles
        self.pose_frame.vertex_colors = pose_frame_copy.vertex_colors

        self.vis.update_geometry(self.current_frame_vis)
        self.vis.update_geometry(self.pose_frame)
        self.refresh_ui(force=True)

    # -----------------------------
    # 主流程
    # -----------------------------
    def run(self):
        if not self.start_stream():
            return
        if not self.load_map(self.map_path):
            self.stop_stream()
            return

        self.setup_visualizer()
        print("\n开始纯 GICP 全局验证：")
        print("- 不使用 ORB")
        print("- 不使用相邻帧配准")
        print("- 每次都直接把当前帧配准到全局地图")
        print("- 按 Q 或 ESC 退出\n")

        try:
            while self.keep_running:
                self.refresh_ui()

                depth_frame, color_frame = self.capture_aligned_frames()
                if depth_frame is None:
                    continue

                self.frame_counter += 1
                if self.frame_counter % self.process_every_n_frames != 0:
                    continue

                current_pcd = self.depth_to_pointcloud(depth_frame, color_frame)
                if len(current_pcd.points) < 100:
                    print("当前点云太稀疏，跳过")
                    continue

                result = self.global_gicp_register(current_pcd)
                if result is None:
                    print("全局 GICP 失败：没有有效候选")
                    continue

                print(
                    f"[Accepted] fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.4f}"
                )

                if result.fitness > self.accept_fitness and result.inlier_rmse < self.accept_rmse:
                    self.last_result = result
                    self.update_visualization(current_pcd, result.transformation)
                else:
                    print("结果未通过接受阈值")

        except KeyboardInterrupt:
            print("\n用户中断，退出。")
        finally:
            if self.vis is not None:
                self.vis.destroy_window()
            self.stop_stream()



def main():
    verifier = GlobalGICPVerifier()
    verifier.run()


if __name__ == "__main__":
    main()
