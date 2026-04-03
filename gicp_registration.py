import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List

import cv2
cv2.setNumThreads(1)
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


class PureGICPRGBLocalization:
    """
    纯 GICP 定位：
    - 不使用 ORB
    - 不做相邻帧里程计配准
    - 未定位或跟踪丢失时：当前帧 -> 全局地图做全局 GICP 搜索
    - 已定位时：当前帧 -> 局部 ROI 地图做局部 GICP 跟踪
    - GICP 后增加 RGB 辅助精修（Open3D 的 Colored ICP）

    说明：Open3D 没有单独的“彩色 GICP”接口，这里实现的是：
    GICP 主配准 + Colored ICP 精修。
    """

    def __init__(self):
        # ----------------
        # 基础配置
        # ----------------
        self.process_every_n_frames = 8
        self.visual_current_pcd_voxel = 0.006
        self.show_live_transformed_cloud = True

        # ----------------
        # Realsense
        # ----------------
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pointcloud = rs.pointcloud()

        self.intrinsics = None
        self.depth_scale = 0.001
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 深度范围
        self.depth_min = 0.07
        self.depth_max = 0.6

        # ----------------
        # GICP / RGB 精修参数
        # ----------------
        self.voxel_size_local = 0.003
        self.local_refine_voxels = [0.018, 0.01, 0.005]
        self.local_refine_iters = [35, 25, 15]
        self.local_fitness_min = 0.35
        self.local_rmse_max = 0.02

        self.enable_rgb_refine = False
        self.rgb_refine_voxels = [0.01, 0.005]
        self.rgb_refine_iters = [15, 10]
        self.rgb_lambda_geometric = 0.968

        # 全局搜索参数
        self.global_voxel_size = 0.03
        self.global_coarse_threshold = 0.06
        self.global_fine_threshold = 0.025
        self.global_final_threshold = 0.012
        self.global_xy_search_range = 0.20
        self.global_xy_step = 0.20
        self.global_yaw_candidates = np.arange(-180, 180, 60)
        self.global_top_k = 1
        self.global_accept_score = 0.05
        self.global_accept_fitness = 0.12
        self.global_accept_rmse = 0.05

        # 点云过滤
        self.current_voxel_pre = 0.005
        self.global_source_voxel = 0.025
        self.map_max_z = 2.0
        self.enable_statistical_filter = True
        self.stat_nb_neighbors = 20
        self.stat_std_ratio = 1.8
        self.enable_radius_filter = True
        self.radius_nb_points = 10
        self.radius = 0.03
        self.min_points_to_process = 40
        self.debug_frame_status = True
        self.worker_sleep = 0.005
        self.max_global_workers = 1
        self.enable_rgb_refine_global = False
        self.global_coarse_max_iter = 8
        self.global_fine_max_iter = 15
        self.global_final_max_iter = 10
        self.live_frame_apply_outlier = False
        self.drop_frames_when_worker_busy = True
        self.relocalize_after_lost_frames = 8
        self.global_retry_counter = 0

        # 已定位后小范围重定位参数
        self.recover_xy_offsets = [-0.18, 0.0, 0.18]
        self.recover_yaw_offsets = [-20, 0, 20]

        # ROI
        self.local_map_radius = 0.9
        self.local_depth_margin = 0.20
        self.local_xy_margin = 0.08

        # 状态机
        self.mode = "GLOBAL"  # GLOBAL / LOCAL
        self.is_localized = False
        self.current_pose = np.eye(4)
        self.last_pose = np.eye(4)
        self.last_good_global_pose = np.eye(4)
        self.consecutive_local_failures = 0
        self.max_local_failures = 2

        # 地图
        self.prior_map_pcd: Optional[o3d.geometry.PointCloud] = None
        self.global_map_down: Optional[o3d.geometry.PointCloud] = None

        # 线程共享数据
        self.state_lock = threading.Lock()
        self.shutdown_flag = False
        self.new_frame_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.latest_input = None
        self.latest_input_id = 0
        self.last_processed_id = 0
        self.latest_status = "等待图像..."
        self.latest_result_pose = np.eye(4)
        self.latest_result_success = False
        self.latest_result_mode = "GLOBAL"
        self.latest_result_score = 0.0
        self.latest_result_fitness = 0.0
        self.latest_result_rmse = 999.0
        self.latest_current_pcd: Optional[o3d.geometry.PointCloud] = None
        self.latest_local_map: Optional[o3d.geometry.PointCloud] = None
        self.latest_color_bgr: Optional[np.ndarray] = None

        # 轨迹
        self.trajectory_points: List[np.ndarray] = []

        # 可视化
        self.vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self.pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self.trajectory = o3d.geometry.LineSet()
        self.vis_current_cloud = o3d.geometry.PointCloud()
        self.vis_local_map = o3d.geometry.PointCloud()
        self._pose_frame_initialized = False
        self._last_pose_for_vis = np.eye(4)

        # 滤波
        self.setup_depth_filters()

    # -----------------------------
    # Realsense / I/O
    # -----------------------------
    def setup_depth_filters(self):
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        self.decimation.set_option(rs.option.filter_magnitude, 1)
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)

    def _apply_depth_filters(self, depth_frame):
        filtered = self.decimation.process(depth_frame)
        filtered = self.spatial.process(filtered)
        filtered = self.temporal.process(filtered)
        filtered = self.hole_filling.process(filtered)
        return filtered

    def start_stream(self) -> bool:
        try:
            profile = self.pipeline.start(self.config)
            print("Realsense stream started.")
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intrinsics = color_stream.get_intrinsics()
            self.camera_matrix = np.array([
                [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
                [0.0, self.intrinsics.fy, self.intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            print(
                f"Intrinsics fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
                f"cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}"
            )
            print(f"Depth scale = {self.depth_scale}")
            return True
        except Exception as e:
            print(f"Failed to start Realsense stream: {e}")
            return False

    def load_prior_map(self, pcd_path: str) -> bool:
        try:
            map_path = os.path.abspath(pcd_path)
            self.prior_map_pcd = o3d.io.read_point_cloud(map_path)
            if len(self.prior_map_pcd.points) == 0:
                raise RuntimeError("地图点云为空")

            if not self.prior_map_pcd.has_colors():
                self.prior_map_pcd.paint_uniform_color([0.7, 0.7, 0.7])

            self.prior_map_pcd = self.filter_pointcloud(self.prior_map_pcd, apply_outlier=True)
            self.prior_map_pcd = self.preprocess_pointcloud(self.prior_map_pcd, self.voxel_size_local, apply_outlier=False)
            self.global_map_down = self.preprocess_pointcloud(self.prior_map_pcd, self.global_voxel_size, apply_outlier=False)
            print("正在预计算全局 GICP 地图...")
            print(f"Prior map loaded from {map_path}.")
            print(f"Map points(local)={len(self.prior_map_pcd.points)}, global={len(self.global_map_down.points)}")
            return True
        except Exception as e:
            print(f"加载先验地图失败: {e}")
            return False

    def capture_frame(self):
        try:
            frames = self.pipeline.poll_for_frames()
        except Exception:
            return None, None
        if not frames:
            return None, None
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_frame = self._apply_depth_filters(depth_frame)
        return depth_frame, color_frame

    def depth_frame_to_pointcloud(self, depth_frame, color_frame) -> o3d.geometry.PointCloud:
        points = self.pointcloud.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        valid_mask = ~np.isnan(vertices).any(axis=1)
        valid_mask &= (vertices[:, 2] > self.depth_min)
        valid_mask &= (vertices[:, 2] < self.depth_max)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices[valid_mask])

        color_image = np.asanyarray(color_frame.get_data())
        color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_flat = color_rgb.reshape(-1, 3)
        colors = color_flat[valid_mask]
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

        pcd = self.filter_pointcloud(pcd, apply_outlier=False, is_live_frame=True)
        if len(pcd.points) > 0 and self.current_voxel_pre > 0:
            pcd = pcd.voxel_down_sample(self.current_voxel_pre)

        if self.live_frame_apply_outlier:
            pcd = self.filter_pointcloud(pcd, apply_outlier=True, is_live_frame=True)
        return pcd

    # -----------------------------
    # 点云预处理 / 配准辅助
    # -----------------------------
    def filter_pointcloud(self, pcd: o3d.geometry.PointCloud, apply_outlier: bool = True, is_live_frame: bool = False) -> o3d.geometry.PointCloud:
        if pcd is None or len(pcd.points) == 0:
            return o3d.geometry.PointCloud()

        pts = np.asarray(pcd.points)
        mask = np.isfinite(pts).all(axis=1)
        mask &= (pts[:, 2] > self.depth_min)
        mask &= (pts[:, 2] < self.depth_max)
        mask &= (np.abs(pts[:, 0]) < max(1.2, self.depth_max * 2.0))
        mask &= (np.abs(pts[:, 1]) < max(1.2, self.depth_max * 2.0))
        if self.map_max_z is not None:
            mask &= (np.abs(pts[:, 2]) < self.map_max_z)

        if not np.any(mask):
            return o3d.geometry.PointCloud()

        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts[mask])
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            out.colors = o3d.utility.Vector3dVector(colors[mask])

        if len(out.points) > 50 and apply_outlier:
            before_count = len(out.points)
            try:
                if self.enable_statistical_filter:
                    nb = min(self.stat_nb_neighbors, max(8, before_count // 20)) if is_live_frame else self.stat_nb_neighbors
                    std_ratio = max(2.2, self.stat_std_ratio) if is_live_frame else self.stat_std_ratio
                    out_filtered, _ = out.remove_statistical_outlier(
                        nb_neighbors=nb, std_ratio=std_ratio
                    )
                    if len(out_filtered.points) >= max(self.min_points_to_process, int(before_count * 0.25)):
                        out = out_filtered
                if self.enable_radius_filter and len(out.points) > 80:
                    nb_pts = min(self.radius_nb_points, max(6, len(out.points) // 40)) if is_live_frame else self.radius_nb_points
                    radius = max(self.radius, 0.04) if is_live_frame else self.radius
                    out_filtered, _ = out.remove_radius_outlier(
                        nb_points=nb_pts, radius=radius
                    )
                    if len(out_filtered.points) >= max(self.min_points_to_process, int(before_count * 0.20)):
                        out = out_filtered
            except Exception:
                pass
        return out

    def preprocess_pointcloud(self, pcd: o3d.geometry.PointCloud, voxel_size: float, apply_outlier: bool = False, is_live_frame: bool = False) -> o3d.geometry.PointCloud:
        base = self.filter_pointcloud(pcd, apply_outlier=apply_outlier, is_live_frame=is_live_frame)
        if len(base.points) == 0:
            return base
        out = base.voxel_down_sample(voxel_size)
        if len(out.points) > 10:
            out.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
            )
        return out

    def make_xyyaw_transform(self, x: float, y: float, yaw_deg: float) -> np.ndarray:
        yaw_rad = np.deg2rad(yaw_deg)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        T = np.eye(4)
        T[:3, :3] = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        T[0, 3] = x
        T[1, 3] = y
        return T

    def crop_local_map_frustum(
        self,
        map_pcd: o3d.geometry.PointCloud,
        init_transform: np.ndarray,
        depth_margin: Optional[float] = None,
        xy_margin: Optional[float] = None,
    ) -> o3d.geometry.PointCloud:
        if depth_margin is None:
            depth_margin = self.local_depth_margin
        if xy_margin is None:
            xy_margin = self.local_xy_margin

        pts_map = np.asarray(map_pcd.points)
        if pts_map.shape[0] == 0:
            return o3d.geometry.PointCloud()

        T_map_to_cam = np.linalg.inv(init_transform)
        pts_h = np.hstack([pts_map, np.ones((pts_map.shape[0], 1), dtype=np.float64)])
        pts_cam = (T_map_to_cam @ pts_h.T).T[:, :3]

        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = pts_cam[:, 2]

        z_min = max(0.01, self.depth_min - 0.02)
        z_max = self.depth_max + depth_margin
        mask = (
            (z > z_min)
            & (z < z_max)
            & (np.abs(x) < z * 1.0 + xy_margin)
            & (np.abs(y) < z * 1.0 + xy_margin)
        )

        local_map = o3d.geometry.PointCloud()
        local_map.points = o3d.utility.Vector3dVector(pts_map[mask])
        if map_pcd.has_colors():
            colors = np.asarray(map_pcd.colors)
            local_map.colors = o3d.utility.Vector3dVector(colors[mask])
        if len(local_map.points) > 10:
            local_map.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size_local * 2.0,
                    max_nn=30,
                )
            )
        return local_map

    def colored_refine(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        init_transform: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        if (not self.enable_rgb_refine) or (not source_pcd.has_colors()) or (not target_pcd.has_colors()):
            return init_transform, 0.0, 999.0

        current_T = init_transform.copy()
        last_fitness = 0.0
        last_rmse = 999.0

        try:
            for voxel, max_iter in zip(self.rgb_refine_voxels, self.rgb_refine_iters):
                source_down = self.preprocess_pointcloud(source_pcd, voxel)
                target_down = self.preprocess_pointcloud(target_pcd, voxel)
                if len(source_down.points) < 30 or len(target_down.points) < 30:
                    continue

                result = o3d.pipelines.registration.registration_colored_icp(
                    source_down,
                    target_down,
                    voxel * 1.5,
                    current_T,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(
                        lambda_geometric=self.rgb_lambda_geometric
                    ),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter,
                    ),
                )
                current_T = result.transformation
                last_fitness = float(result.fitness)
                last_rmse = float(result.inlier_rmse)
        except Exception as e:
            print(f"Colored ICP refine failed: {e}")
        return current_T, last_fitness, last_rmse

    def local_registration(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        init_transform: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float]:
        current_T = init_transform.copy()
        gicp_fitness = 0.0
        gicp_rmse = 999.0

        try:
            for voxel, max_iter in zip(self.local_refine_voxels, self.local_refine_iters):
                source_down = self.preprocess_pointcloud(source_pcd, voxel)
                target_down = self.preprocess_pointcloud(target_pcd, voxel)
                if len(source_down.points) < 30 or len(target_down.points) < 30:
                    continue

                result = o3d.pipelines.registration.registration_generalized_icp(
                    source_down,
                    target_down,
                    voxel * 2.0,
                    current_T,
                    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=max_iter,
                    ),
                )
                current_T = result.transformation
                gicp_fitness = float(result.fitness)
                gicp_rmse = float(result.inlier_rmse)

            rgb_fitness = 0.0
            if self.enable_rgb_refine:
                current_T, rgb_fitness, _ = self.colored_refine(source_pcd, target_pcd, current_T)

            score = float(gicp_fitness - 3.0 * gicp_rmse + 0.1 * rgb_fitness)
            return current_T, score, gicp_fitness, gicp_rmse
        except Exception as e:
            print(f"Local registration failed: {e}")
            return init_transform.copy(), -1e9, 0.0, 999.0

    # -----------------------------
    # 全局 GICP
    # -----------------------------
    def get_global_translation_candidates(self) -> List[Tuple[float, float]]:
        if self.is_localized:
            cx = float(self.current_pose[0, 3])
            cy = float(self.current_pose[1, 3])
            return [
                (cx + dx, cy + dy)
                for dx in self.recover_xy_offsets
                for dy in self.recover_xy_offsets
            ]

        r = self.global_xy_search_range
        s = self.global_xy_step
        xs = np.arange(-r, r + 1e-6, s)
        ys = np.arange(-r, r + 1e-6, s)
        return [(float(x), float(y)) for x in xs for y in ys]

    def get_global_yaw_candidates(self) -> List[float]:
        if self.is_localized:
            yaw = np.rad2deg(np.arctan2(self.current_pose[1, 0], self.current_pose[0, 0]))
            return [float(yaw + dyaw) for dyaw in self.recover_yaw_offsets]
        return [float(v) for v in self.global_yaw_candidates]

    def _evaluate_global_candidate(self, source_global, target_global, x, y, yaw_deg):
        init_T = self.make_xyyaw_transform(x, y, yaw_deg)
        try:
            coarse_res = o3d.pipelines.registration.registration_generalized_icp(
                source_global,
                target_global,
                self.global_coarse_threshold,
                init_T,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-5,
                    relative_rmse=1e-5,
                    max_iteration=self.global_coarse_max_iter,
                ),
            )
        except Exception:
            return None

        if coarse_res.fitness < 0.05:
            return None

        score = float(coarse_res.fitness - 3.0 * coarse_res.inlier_rmse)
        return (score, coarse_res, x, y, yaw_deg)

    def global_registration(self, current_pcd: o3d.geometry.PointCloud) -> Tuple[Optional[np.ndarray], float, float, float, Optional[o3d.geometry.PointCloud]]:
        """
        使用验证脚本同款的全局 GICP：
        粗搜索 top-K -> 全局 fine -> 直接对整张全局地图做最终精修。
        支持并行 coarse 候选评估，并对输入点云先做离群点和视距过滤。
        """
        source_global = self.preprocess_pointcloud(current_pcd, self.global_source_voxel, apply_outlier=False, is_live_frame=True)
        target_global = self.global_map_down
        if target_global is None or len(source_global.points) < 40 or len(target_global.points) < 30:
            return None, -1e9, 0.0, 999.0, None

        xy_candidates = self.get_global_translation_candidates()
        yaw_candidates = self.get_global_yaw_candidates()
        candidate_params = [(x, y, yaw_deg) for x, y in xy_candidates for yaw_deg in yaw_candidates]

        coarse_pool = []
        if self.max_global_workers <= 1:
            for x, y, yaw_deg in candidate_params:
                if self.shutdown_flag:
                    return None, -1e9, 0.0, 999.0, None
                res = self._evaluate_global_candidate(source_global, target_global, x, y, yaw_deg)
                if res is not None:
                    coarse_pool.append(res)
        else:
            with ThreadPoolExecutor(max_workers=self.max_global_workers) as ex:
                futures = [
                    ex.submit(self._evaluate_global_candidate, source_global, target_global, x, y, yaw_deg)
                    for x, y, yaw_deg in candidate_params
                ]
                for fut in as_completed(futures):
                    if self.shutdown_flag:
                        return None, -1e9, 0.0, 999.0, None
                    res = fut.result()
                    if res is not None:
                        coarse_pool.append(res)

        if not coarse_pool:
            return None, -1e9, 0.0, 999.0, None

        coarse_pool.sort(key=lambda item: item[0], reverse=True)
        coarse_pool = coarse_pool[: self.global_top_k]

        source_local = self.preprocess_pointcloud(current_pcd, self.voxel_size_local, apply_outlier=False, is_live_frame=True)
        target_local = self.prior_map_pcd
        if target_local is None or len(source_local.points) < 40 or len(target_local.points) < 50:
            return None, -1e9, 0.0, 999.0, None

        best_T = None
        best_score = -1e9
        best_fitness = 0.0
        best_rmse = 999.0

        for _, coarse_res, x, y, yaw_deg in coarse_pool:
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
                        max_iteration=self.global_fine_max_iter,
                    ),
                )

                final_res = o3d.pipelines.registration.registration_generalized_icp(
                    source_local,
                    target_local,
                    self.global_final_threshold,
                    fine_res.transformation,
                    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-7,
                        relative_rmse=1e-7,
                        max_iteration=self.global_final_max_iter,
                    ),
                )

                final_T = final_res.transformation
                final_fitness = float(final_res.fitness)
                final_rmse = float(final_res.inlier_rmse)

                score = float(final_fitness - 3.0 * final_rmse)
                print(
                    f"[GLOBAL GICP] x0={x:+.2f}, y0={y:+.2f}, yaw0={yaw_deg:+4.0f} | "
                    f"score={score:.3f}, fit={final_fitness:.3f}, rmse={final_rmse:.4f}"
                )

                if score > best_score:
                    best_T = final_T
                    best_score = score
                    best_fitness = final_fitness
                    best_rmse = final_rmse
            except Exception:
                continue

        return best_T, best_score, best_fitness, best_rmse, None

    # -----------------------------
    # 定位状态机（后台线程）
    # -----------------------------
    def localization_worker_loop(self):
        while not self.shutdown_flag:
            got_frame = self.new_frame_event.wait(timeout=0.1)
            if self.shutdown_flag:
                break
            if not got_frame:
                continue

            with self.state_lock:
                if self.latest_input_id == self.last_processed_id or self.latest_input is None:
                    self.new_frame_event.clear()
                    continue
                input_id = self.latest_input_id
                current_pcd = self.latest_input
                self.new_frame_event.clear()

            success = False
            score = -1e9
            fitness = 0.0
            rmse = 999.0
            out_pose = self.current_pose.copy()
            out_local_map = None
            out_mode = self.mode
            status = ""

            try:
                if (not self.is_localized) or self.mode == "GLOBAL":
                    out_mode = "GLOBAL"
                    should_run_global = (self.global_retry_counter == 0) or (self.global_retry_counter >= self.relocalize_after_lost_frames)
                    if should_run_global:
                        T, score, fitness, rmse, out_local_map = self.global_registration(current_pcd)
                        self.global_retry_counter = 0
                        if (
                            T is not None
                            and score > self.global_accept_score
                            and fitness > self.global_accept_fitness
                            and rmse < self.global_accept_rmse
                        ):
                            out_pose = T
                            success = True
                            self.current_pose = T
                            self.last_good_global_pose = T.copy()
                            self.is_localized = True
                            self.mode = "LOCAL"
                            self.consecutive_local_failures = 0
                            self.global_retry_counter = 0
                            status = f"全局初始化成功 score={score:.3f} fit={fitness:.3f} rmse={rmse:.4f}"
                        else:
                            self.is_localized = False
                            self.mode = "GLOBAL"
                            status = f"全局初始化失败 score={score:.3f} fit={fitness:.3f} rmse={rmse:.4f}"
                    else:
                        self.global_retry_counter += 1
                        self.is_localized = False
                        self.mode = "GLOBAL"
                        status = f"跳过全局重定位 {self.global_retry_counter}/{self.relocalize_after_lost_frames}"
                else:
                    out_mode = "LOCAL"
                    local_map = self.crop_local_map_frustum(self.prior_map_pcd, self.current_pose)
                    out_local_map = local_map
                    if len(local_map.points) < 100:
                        self.consecutive_local_failures += 1
                        success = False
                        status = "局部地图点太少，回到全局模式"
                    else:
                        T, score, fitness, rmse = self.local_registration(current_pcd, local_map, self.current_pose)
                        if fitness > self.local_fitness_min and rmse < self.local_rmse_max:
                            out_pose = T
                            success = True
                            self.current_pose = T
                            status = f"局部跟踪成功 score={score:.3f} fit={fitness:.3f} rmse={rmse:.4f}"
                            self.consecutive_local_failures = 0
                        else:
                            self.consecutive_local_failures += 1
                            success = False
                            status = f"局部跟踪失败({self.consecutive_local_failures}) fit={fitness:.3f} rmse={rmse:.4f}"

                    if self.consecutive_local_failures >= self.max_local_failures:
                        self.is_localized = False
                        self.mode = "GLOBAL"
                        self.global_retry_counter = 0
                        status += " -> 切换到全局重定位"

            except Exception as e:
                success = False
                status = f"后台定位异常: {e}"
                self.is_localized = False
                self.mode = "GLOBAL"

            with self.state_lock:
                self.last_processed_id = input_id
                self.latest_result_pose = out_pose.copy()
                self.latest_result_success = success
                self.latest_result_mode = self.mode
                self.latest_result_score = score
                self.latest_result_fitness = fitness
                self.latest_result_rmse = rmse
                self.latest_status = status
                self.latest_current_pcd = current_pcd
                self.latest_local_map = out_local_map
                if success:
                    if len(self.trajectory_points) == 0 or not np.allclose(self.trajectory_points[-1], out_pose[:3, 3]):
                        self.trajectory_points.append(out_pose[:3, 3].copy())

    # -----------------------------
    # 可视化（主线程，Open3D 3D 交互窗口）
    # -----------------------------
    def _on_quit(self):
        self.shutdown_flag = True
        self.new_frame_event.set()
        return False

    def _save_trajectory_to_file(self):
        with self.state_lock:
            traj_pts = [p.copy() for p in self.trajectory_points]

        if len(traj_pts) == 0:
            print("\n当前没有可保存的相机轨迹")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(base_dir, f"camera_trajectory_{timestamp}.txt")

        traj = np.asarray(traj_pts, dtype=np.float64)
        header = "# index x y z"
        data = np.column_stack([np.arange(len(traj), dtype=np.int32), traj])
        np.savetxt(save_path, data, fmt=["%d", "%.6f", "%.6f", "%.6f"], header=header, comments="")
        print(f"\n相机轨迹已保存: {save_path}")

    def _on_save_trajectory(self):
        try:
            self._save_trajectory_to_file()
        except Exception as e:
            print(f"\n保存相机轨迹失败: {e}")
        return False

    def _center_of_map(self) -> np.ndarray:
        if self.prior_map_pcd is None or len(self.prior_map_pcd.points) == 0:
            return np.zeros(3, dtype=np.float64)
        pts = np.asarray(self.prior_map_pcd.points)
        return pts.mean(axis=0)

    def init_visualization(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="GICP 3D Viewer", width=1440, height=900, visible=True)
        self.vis.register_key_callback(ord("Q"), lambda vis: self._on_quit())
        self.vis.register_key_callback(ord("S"), lambda vis: self._on_save_trajectory())
        self.vis.register_key_callback(256, lambda vis: self._on_quit())  # ESC

        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.array([0.05, 0.05, 0.05], dtype=np.float64)
        render_opt.point_size = 2.0
        render_opt.line_width = 3.0

        map_copy = o3d.geometry.PointCloud(self.prior_map_pcd)
        if not map_copy.has_colors():
            map_copy.paint_uniform_color([0.65, 0.65, 0.65])
        self.vis.add_geometry(map_copy)

        self.vis_current_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.vis_current_cloud)

        self.trajectory = o3d.geometry.LineSet()
        self.vis.add_geometry(self.trajectory)

        self.pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
        self.vis.add_geometry(self.pose_frame)
        self._pose_frame_initialized = True
        self._last_pose_for_vis = np.eye(4)

        bbox = self.prior_map_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = np.maximum(bbox.get_extent(), np.array([0.3, 0.3, 0.3]))
        ctr = self.vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front(np.array([0.0, -1.0, -0.35]))
        ctr.set_up(np.array([0.0, 0.0, 1.0]))
        ctr.set_zoom(float(np.clip(0.9 / np.max(extent), 0.08, 0.7)))

        self.vis.poll_events()
        self.vis.update_renderer()

    def build_trajectory_lineset(self, traj_pts: List[np.ndarray]) -> o3d.geometry.LineSet:
        lineset = o3d.geometry.LineSet()
        if len(traj_pts) == 0:
            return lineset
        pts = np.asarray(traj_pts, dtype=np.float64)
        lineset.points = o3d.utility.Vector3dVector(pts)
        if len(traj_pts) >= 2:
            lines = np.array([[i, i + 1] for i in range(len(traj_pts) - 1)], dtype=np.int32)
            colors = np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float64), (len(lines), 1))
            lineset.lines = o3d.utility.Vector2iVector(lines)
            lineset.colors = o3d.utility.Vector3dVector(colors)
        return lineset

    def update_visualization(self):
        if self.vis is None:
            return

        with self.state_lock:
            pose = self.latest_result_pose.copy()
            traj_pts = [p.copy() for p in self.trajectory_points]
            current_pcd = None if self.latest_current_pcd is None else o3d.geometry.PointCloud(self.latest_current_pcd)
            status = self.latest_status
            mode = self.latest_result_mode
            fitness = float(self.latest_result_fitness)
            rmse = float(self.latest_result_rmse)
            score = float(self.latest_result_score)

        if current_pcd is not None and len(current_pcd.points) > 0:
            current_pcd.transform(pose)
            if self.visual_current_pcd_voxel > 0 and len(current_pcd.points) > 0:
                current_pcd = current_pcd.voxel_down_sample(self.visual_current_pcd_voxel)
            if not current_pcd.has_colors():
                current_pcd.paint_uniform_color([0.1, 0.8, 1.0])
            self.vis_current_cloud.points = current_pcd.points
            if current_pcd.has_colors():
                self.vis_current_cloud.colors = current_pcd.colors
            else:
                self.vis_current_cloud.paint_uniform_color([0.1, 0.8, 1.0])
        else:
            self.vis_current_cloud.clear()

        new_traj = self.build_trajectory_lineset(traj_pts)
        self.trajectory.points = new_traj.points
        self.trajectory.lines = new_traj.lines
        self.trajectory.colors = new_traj.colors

        delta = pose @ np.linalg.inv(self._last_pose_for_vis)
        self.pose_frame.transform(delta)
        self._last_pose_for_vis = pose.copy()

        self.vis.update_geometry(self.vis_current_cloud)
        self.vis.update_geometry(self.trajectory)
        self.vis.update_geometry(self.pose_frame)
        self.vis.poll_events()
        self.vis.update_renderer()
        try:
            self.vis.get_view_control().set_constant_z_far(1000.0)
        except Exception:
            pass
        print(f"\r[{mode}] score={score:.3f} fit={fitness:.3f} rmse={rmse:.4f} | {status[:80]} | Q退出 S保存轨迹", end="", flush=True)

    # -----------------------------
    # 主循环
    # -----------------------------
    def start_worker(self):
        self.worker_thread = threading.Thread(target=self.localization_worker_loop, daemon=True)
        self.worker_thread.start()

    def run(self):
        if self.prior_map_pcd is None:
            raise RuntimeError("请先加载先验地图")

        self.init_visualization()
        self.start_worker()

        frame_counter = 0
        try:
            while not self.shutdown_flag:
                frame_counter += 1
                depth_frame, color_frame = self.capture_frame()
                if depth_frame is None or color_frame is None:
                    self.update_visualization()
                    time.sleep(0.01)
                    continue

                if frame_counter == 1 or frame_counter % self.process_every_n_frames == 0:
                    with self.state_lock:
                        worker_busy = self.latest_input_id != self.last_processed_id

                    if self.drop_frames_when_worker_busy and worker_busy:
                        with self.state_lock:
                            if self.debug_frame_status:
                                self.latest_status = f"后台忙，跳过当前帧, mode={self.mode}"
                    else:
                        current_pcd = self.depth_frame_to_pointcloud(depth_frame, color_frame)
                        cur_pts = len(current_pcd.points)
                        if cur_pts >= self.min_points_to_process:
                            with self.state_lock:
                                self.latest_input = current_pcd
                                self.latest_input_id += 1
                                self.latest_current_pcd = current_pcd
                                if self.debug_frame_status:
                                    self.latest_status = f"收到图像: {cur_pts} pts, mode={self.mode}"
                            self.new_frame_event.set()
                        else:
                            with self.state_lock:
                                self.latest_current_pcd = current_pcd
                                self.latest_status = f"当前图像点太少: {cur_pts} pts"

                self.update_visualization()
                time.sleep(self.worker_sleep)
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            print()
            self.shutdown_flag = True
            self.new_frame_event.set()
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=1.0)
            try:
                self.pipeline.stop()
            except Exception:
                pass
            if self.vis is not None:
                self.vis.destroy_window()
                self.vis = None


def main():
    localizer = PureGICPRGBLocalization()
    if not localizer.start_stream():
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prior_map_path = os.path.join(base_dir, "my_environment_map_pgo.ply")
    if not localizer.load_prior_map(prior_map_path):
        localizer.pipeline.stop()
        return

    localizer.run()


if __name__ == "__main__":
    main()
