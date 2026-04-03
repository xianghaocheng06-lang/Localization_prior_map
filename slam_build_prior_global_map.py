import tempfile
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import copy
import time
import imageio.v3 as iio
from collections import deque
import os
import shutil

class PointCloudMapBuilder:
    """
    优化版点云地图构建工具
    1. 使用 Open3D Native C++ TSDF (ScalableTSDFVolume) 实现实时融合。
    2. 采用 Frame-to-Model (Tracking Map) 策略减少累积误差。
    3. 引入关键帧机制，保障长时间运行的效率。
    4. 新增完整的位姿图优化（PGO）和全局一致性重融合。
    """
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.align = rs.align(rs.stream.color)
        self.pointcloud = rs.pointcloud()

        # --- TSDF Parameters (Open3D Native) ---
        self.voxel_size_map = 0.003  # (根据性能调整，太小会增加显存/内存消耗)
        self.sdf_trunc = 0.03        # 截断距离 (通常设为 voxel_size 的 5-10 倍)

       # 使用 ScalableTSDFVolume，它使用哈希表存储体素，只存储有物体的地方，内存效率高且速度快
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size_map,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # --- 追踪专用地图 (Tracking Map) ---
        self.tracking_map = o3d.geometry.PointCloud() 
        self.last_keyframe_pose = np.eye(4) 

        # --- PGO 位姿图优化 ---
        # 存储所有关键帧的元数据：(id, pgo_id, pose, pcd_down, orb_features, rgbd)
        self.keyframes = [] 
        self.pose_graph = o3d.pipelines.registration.PoseGraph() # 存储节点和边
        self.frame_id_counter = 0 # 也是 PoseGraphNode 的索引

        self.last_loop_detection_time = time.time()
        self.loop_detection_interval = 5.0 # 每 5 秒检测一次回环

        # 全局点云地图 (仅用于可视化缓存，从TSDF提取)
        self.vis_map = o3d.geometry.PointCloud()
        self.global_transform = np.eye(4) # 当前相机的世界坐标系位姿

        self.global_orb_points = []
        self.global_orb_descriptors = []
        self.frame_id_counter = 0

        # 滑动窗口
        self.window_size = 20
        self.sliding_window = deque(maxlen=self.window_size)

        self.last_pcd = None
        self.global_transform = np.eye(4)

        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=10, edgeThreshold=31)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # 用于匹配ORB描述符

        # 回环RANSAC参数
        self.loop_ransac_distance = 0.05 # RANSAC几何验证的内点距离阈值 (5cm)

        # 点云处理参数
        self.voxel_size_icp = 0.005
        self.coarse_threshold = self.voxel_size_icp * 10
        self.fine_threshold = self.voxel_size_icp

        # 深度范围（D405 适用于近距离）
        self.depth_min = 0.07
        self.depth_max = 0.6

        # 深度滤波器初始化
        self.setup_depth_filters()

        # 相机内参和配置文件（启动时获取）
        self.profile = None

        # --- GIF/可视化相关 ---
        self.vis = None
        self.frame_paths = []  # 存储用于生成GIF的图片路径
        self.temp_dir = None
        self.map_geometry = o3d.geometry.PointCloud()
        self.initial_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.pose_frame = copy.deepcopy(self.initial_pose_frame)  # 实际在可视化器中更新的几何体
        self.trajectory_points = [np.array([0, 0, 0])]
        self.trajectory = o3d.geometry.LineSet()

        # 标志用于判断可视化器中是否已加入几何体
        self._vis_has_map = False
        self._vis_has_pose = False
        self._vis_has_trajectory = False
    
    # @staticmethod
    # def tsdf_fusion_update(D_old: float, W_old: float, d_new: float, w_new: float, max_weight: float) -> tuple[float, float]:
    #     """
    #     Implementation of the TSDF fusion update with max weight clamping.
    #     (Logic adapted from tsdf_fusion.py to fit class structure)
        
    #     Args:
    #         D_old: Old normalized TSDF value ([-1, 1]).
    #         W_old: Old weight.
    #         d_new: New normalized TSDF measurement ([-1, 1]).
    #         w_new: New observation weight (e.g., 1.0).
    #         max_weight: Maximum allowed weight for clamping.

    #     Returns:
    #         tuple[float, float]: (D_new, W_new) Fused TSDF value and new weight.
    #     """
    #     total_weight = W_old + w_new
        
    #     if total_weight == 0:
    #         return d_new, w_new 
        
    #     # 1. Fuse distance value (D_new) using weighted average
    #     D_new = (W_old * D_old + w_new * d_new) / total_weight
        
    #     # 2. Fuse weight (W_new) and clamp to max_weight
    #     W_new = min(total_weight, max_weight)
        
    #     return D_new, W_new


    def start_camera(self):
        """启动相机""" 
        try:
            profile = self.pipeline.start(self.config)
            self.profile = profile

            # 获取内参
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            
            # 转换为 Open3D 的 PinholeCameraIntrinsic 对象
            self.depth_intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
            )

            # 获取深度比例尺和内参
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

            print(f"相机启动成功, 深度比例: {self.depth_scale}")
            return True
        except Exception as e:
            print(f"相机启动失败: {e}")
            return False

    def setup_depth_filters(self):
        """初始化深度滤波器"""
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        # 设置滤波器参数
        self.decimation.set_option(rs.option.filter_magnitude, 1)
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)

    def _apply_depth_filters(self, depth_frame):
        """深度图滤波链"""
        filtered = self.decimation.process(depth_frame)
        filtered = self.spatial.process(filtered) 
        filtered = self.temporal.process(filtered)
        # filtered = self.hole_filling.process(filtered)
        return filtered

    def _add_color_to_pointcloud(self, pcd, color_frame, valid_mask):
        """将颜色映射到点云"""
        color_image = np.asanyarray(color_frame.get_data())
        flat_colors = color_image.reshape(-1, 3)
        if len(flat_colors) >= len(valid_mask):
            colors = flat_colors[valid_mask]
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    def capture_frame_data(self):
        """捕获数据，返回 Open3D 需要的格式"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        if not depth or not color:
            return None

        depth = self._apply_depth_filters(depth)

        # 转换为 Numpy
        color_image = np.asanyarray(color.get_data())
        depth_image = np.asanyarray(depth.get_data())

        cv2.imshow("构建地图 - 按'c'捕获", color_image)

        # 生成 Open3D 图像对象
        o3d_color = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        o3d_depth = o3d.geometry.Image(depth_image)

        # 生成 RGBD 图像 (用于 TSDF 和 ICP)
        # convert_rgb_to_intensity=False 也就是保持彩色
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, 
            depth_scale=1.0/self.depth_scale, 
            depth_trunc=self.depth_max, 
            convert_rgb_to_intensity=False
        )

        # 同时也生成点云用于 ORB 和可视化 (如果需要单帧)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.depth_intrinsics_o3d
        )
        
        return pcd, rgbd_image, color_image, depth_image

    def extract_orb(self, color_img, depth_img):
        """提取 ORB 特征 (保持你原有的逻辑，稍作优化)"""
        gray_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
        
        orb_points_3d = []
        orb_descriptors_filtered = []

        if descriptors is not None:
            # 向量化处理会更快，但为了保持逻辑清晰，这里只做简单的边界检查
            for kp, desc in zip(keypoints, descriptors):
                u, v = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= v < depth_img.shape[0] and 0 <= u < depth_img.shape[1]:
                    d = depth_img[v, u] * self.depth_scale
                    if self.depth_min < d < self.depth_max:
                        # 简单的针孔模型反投影
                        z = d
                        x = (u - self.depth_intrinsics_o3d.intrinsic_matrix[0, 2]) * z / self.depth_intrinsics_o3d.intrinsic_matrix[0, 0]
                        y = (v - self.depth_intrinsics_o3d.intrinsic_matrix[1, 2]) * z / self.depth_intrinsics_o3d.intrinsic_matrix[1, 1]
                        orb_points_3d.append([x, y, z])
                        orb_descriptors_filtered.append(desc)
        
        return np.array(orb_points_3d), np.array(orb_descriptors_filtered)
    
    def integrate_tsdf(self, rgbd_image, T_camera_to_world):
        """
        核心优化：使用 C++ 绑定的 TSDF 积分
        T_camera_to_world: 相机在世界坐标系下的位姿 (Pose)
        """
        # Open3D 的 integrate 函数通常需要传入 "Extrinsic" (World-to-Camera)
        # 即 Pose 的逆矩阵
        T_world_to_camera = np.linalg.inv(T_camera_to_world)
        
        self.tsdf_volume.integrate(
            rgbd_image,
            self.depth_intrinsics_o3d,
            T_world_to_camera
        )
    
    def is_keyframe(self, current_pose):
        """检查是否需要作为关键帧更新地图"""
        # 计算相对运动
        T_delta = np.linalg.inv(self.last_keyframe_pose) @ current_pose
        trans_dist = np.linalg.norm(T_delta[:3, 3])
        rotation_matrix = T_delta[:3, :3]
        trace = np.trace(rotation_matrix)
        trace = np.clip((trace - 1) / 2, -1.0, 1.0)
        rot_angle = np.arccos(trace)

        # 阈值：移动 8cm 或 旋转 5度 (~0.08弧度)
        if trans_dist > 0.05 or abs(rot_angle) > 0.08: 
            return True
        return False
    
    def add_keyframe(self, pcd_down, orb_points_3d, orb_descriptors, rgbd_image):
        """将当前帧作为关键帧添加到列表中，并添加 Odometry 边到 PGO"""
        if orb_descriptors.shape[0] < 50: return # 特征点太少，不作为关键帧

        new_node_index = len(self.keyframes)
        
        # 1. 添加 Pose Graph 节点
        # 使用当前全局位姿作为初始估计
        self.pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(self.global_transform)
        )
        
        # 2. 如果不是第一帧，添加 Odometry 边 (连接相邻关键帧)
        if new_node_index > 0:
            last_node_index = new_node_index - 1
            last_keyframe_pose = self.keyframes[-1]['pose']
            
            # T_relative = T_last_to_current = (T_last_world)^-1 @ T_current_world
            T_relative = np.linalg.inv(last_keyframe_pose) @ self.global_transform

            # Odometry 约束置信度高，使用高权重信息矩阵
            information = np.identity(6) * 100.0
            
            self.pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_node_id=last_node_index,
                    target_node_id=new_node_index,
                    transformation=T_relative,
                    information=information,
                    uncertain=False # Odometry 边默认不是不确定的
                )
            )

        # 3. 存储关键帧数据
        self.keyframes.append({
            'id': self.frame_id_counter,
            'pgo_id': new_node_index,
            'pose': copy.deepcopy(self.global_transform), # 原始位姿 (未优化)
            'pcd_down': pcd_down,
            'points_3d': orb_points_3d,
            'descriptors': orb_descriptors,
            'rgbd': rgbd_image # 存储 RGBD 图像用于最终重融合
        })

        # 新增：累计全局 ORB 数据
        if orb_points_3d is not None and len(orb_points_3d) > 0:
            pose = self.global_transform
            R = pose[:3, :3]
            t = pose[:3, 3]

            pts_world = (R @ orb_points_3d.T).T + t
            frame_ids = np.full((pts_world.shape[0], 1), self.frame_id_counter)
            pts_with_id = np.hstack([pts_world, frame_ids])

            self.global_orb_points.extend(pts_with_id.tolist())
            self.global_orb_descriptors.extend(orb_descriptors.tolist())

        self.frame_id_counter += 1
    
    def _add_loop_constraint(self, source_idx, target_idx, transformation, information):
        """
        添加回环约束边到位姿图。
        transformation: T_source_to_target
        """
        self.pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=source_idx,
                target_node_id=target_idx,
                transformation=transformation,
                information=information,
                uncertain=True # 回环约束是不确定的，权重交给优化器
            )
        )
    
    def _detect_and_correct_loop(self, current_keyframe):
        """
        回环检测：找到回环，计算约束，添加到位姿图
        """
        current_time = time.time()
        if current_time - self.last_loop_detection_time < self.loop_detection_interval:
            return

        print("\n--- 正在进行回环检测 ---")
        self.last_loop_detection_time = current_time

        current_desc = current_keyframe['descriptors']
        
        current_pgo_id = current_keyframe['pgo_id']
        # 仅搜索非相邻的历史帧
        min_search_idx = max(0, current_pgo_id - 20) 

        best_loop_idx = -1
        best_inlier_count = 0
        loop_transform = np.eye(4)
        best_information = np.identity(6)

        for i in range(min_search_idx):
            past_keyframe = self.keyframes[i]
            past_desc = past_keyframe['descriptors']

            # 1. ORB 特征匹配 (粗匹配)
            if past_desc.shape[0] < 50: continue

            matches = self.bf_matcher.match(current_desc, past_desc)
            if len(matches) < 20: continue # 匹配点太少，忽略

            # 提取匹配点的3D坐标
            src_pts = np.asarray([current_keyframe['points_3d'][m.queryIdx] for m in matches])
            tgt_pts = np.asarray([past_keyframe['points_3d'][m.trainIdx] for m in matches])
            
            source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts))
            target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts))
            
            # 2. RANSAC 几何验证 (精匹配)
            # RANSAC 结果是 T_current_to_past
            ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                source_pcd, target_pcd, o3d.utility.Vector2iVector([[i, i] for i in range(len(matches))]),
                self.loop_ransac_distance,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, # 3点是最小集合
                [],
                o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=1000, confidence=0.999)
            )

            if ransac_result.inlier_rmse < 0.02 and ransac_result.fitness > best_inlier_count:
                best_inlier_count = ransac_result.fitness
                best_loop_idx = i
                # T_current_to_past
                loop_transform = ransac_result.transformation 
                
                # 3. 计算信息矩阵 (Loop Closure Information)
                # 简单启发式：根据 RANSAC 的 Fitness 确定置信度
                confidence = max(0.01, ransac_result.fitness) 
                # 平移和旋转的权重差异，鼓励平移更精确
                information = np.identity(6) * (confidence * 50.0)
                information[3:, 3:] *= 2.0 
                best_information = information

        if best_loop_idx != -1:
            # 找到回环
            past_keyframe = self.keyframes[best_loop_idx]
            
            print(f"!!! 回环检测成功 !!! 帧 {current_keyframe['id']} 匹配到帧 {past_keyframe['id']}")
            print(f"RANSAC Inlier Fitness: {best_inlier_count:.3f}")
            
            # 4. 计算相对变换 T_past_to_current = T_current_to_past^-1
            T_past_to_current = np.linalg.inv(loop_transform)
            
            # 5. 添加回环边到位姿图
            self._add_loop_constraint(
                source_idx=past_keyframe['pgo_id'],
                target_idx=current_keyframe['pgo_id'],
                transformation=T_past_to_current,
                information=best_information
            )
        else:
            print("未检测到有效回环。")

    def optimize_global_map(self, output_file):
        """
        执行全局位姿图优化 (PGO)，并使用优化后的位姿重构 TSDF 地图。
        """
        if len(self.keyframes) < 2:
            print("关键帧不足，跳过全局优化。")
            final_pcd = self.tsdf_volume.extract_point_cloud()
            o3d.io.write_point_cloud(output_file, final_pcd)
            print(f"已保存未经优化的地图: {output_file}")
            return
            
        print("\n--- 正在执行位姿图优化 (PGO) ---")
        
        # 优化配置
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size_icp * 1.5,
            edge_prune_threshold=0.25, # 剪枝高误差边
            reference_node=0, # 固定第一个节点
        )

        # 2. FIX: 收敛标准 (Criteria) - 创建对象后设置属性
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        criteria.max_iteration = 500
        
        # 执行优化
        o3d.pipelines.registration.global_optimization(
            self.pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            criteria, # <-- 修复: 插入 GlobalOptimizationConvergenceCriteria
            option
        )
        
        print("--- PGO 优化完成，开始全局一致性重融合 ---")

        # 1. 重置 TSDF Volume
        self.tsdf_volume.reset()
        
        # 2. 使用优化后的位姿进行重融合
        for i, node in enumerate(self.pose_graph.nodes):
            kf_data = self.keyframes[i]
            # 获取优化后的位姿 T_world
            optimized_pose = node.pose
            
            if kf_data['rgbd'] is not None:
                # 重新融合
                self.integrate_tsdf(kf_data['rgbd'], optimized_pose)
            else:
                print(f"警告: 关键帧 {kf_data['id']} 缺少 RGBD 数据，跳过融合。")
        
        print("--- 重融合完成，生成最终地图 ---")
        final_pcd = self.tsdf_volume.extract_point_cloud()
        o3d.io.write_point_cloud(output_file, final_pcd)
        print(f"已保存全局优化后的地图: {output_file}")
    
    def update_tracking_map(self, new_pcd, pose):
        """更新轻量级追踪地图"""
        # 将当前帧变换到世界坐标系
        pcd_world = copy.deepcopy(new_pcd).transform(pose)
        self.tracking_map += pcd_world
        
        # 关键优化：时刻保持追踪地图稀疏 (降采样)
        self.tracking_map = self.tracking_map.voxel_down_sample(self.voxel_size_icp)
        
        # 必须重新计算法线，否则 Point-to-Plane ICP 会失效
        self.tracking_map.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_icp * 2, max_nn=30)
        )
        self.last_keyframe_pose = pose


    def preprocess_pointcloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """点云去噪 + 下采样 + 法线估计"""
        if len(pcd.points) < 100:
            return pcd

        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size_icp)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_icp * 2.0, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        return pcd

    def get_local_map(self):
        """从滑动窗口构建局部地图用于配准"""
        local_map = o3d.geometry.PointCloud()
        for pcd in self.sliding_window:
            local_map += pcd

        return local_map.voxel_down_sample(self.voxel_size_icp)

    def robust_icp_registration(self, source, target):
        """鲁棒的ICP配准 - 结合多级ICP和Colored ICP（若可用）"""
        initial_transform = self.global_transform
        result_icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, self.coarse_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )

        result_icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, self.fine_threshold, result_icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )

        try:
            result_icp_fine = o3d.pipelines.registration.registration_colored_icp(
                source, target, self.voxel_size_icp, result_icp_fine.transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.5),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30)
            )
        except Exception:
            pass

        return result_icp_fine

    # --- 新增可视化方法（改进） ---
    def setup_o3d_visualizer(self, window_name="实时地图构建"):
        """初始化Open3D可视化器并加入初始几何体：
        - 全局固定栅格（grid）作为参考框架
        - 全局点云（map_geometry）用于不断更新点云
        - 小尺寸相机坐标系（pose_frame）表示相机位姿
        - 相机轨迹（trajectory）用于显示移动路径
        视角固定为全局远景，保证栅格和地图同时可见。
        """
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=1280, height=720)

        # 全局点云几何体（用于更新）
        self.map_geometry = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.map_geometry)
        self._vis_has_map = True

        # 小尺寸pose框（用于表示相机）
        # 用比默认更小的坐标系，便于在大范围地图中可见
        self.initial_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        self.pose_frame = copy.deepcopy(self.initial_pose_frame)
        self.vis.add_geometry(self.pose_frame)
        self._vis_has_pose = True

        # 轨迹 LineSet
        self.trajectory = o3d.geometry.LineSet()
        self.vis.add_geometry(self.trajectory)
        self._vis_has_trajectory = True

        # 创建一个大范围的 XY 栅格，固定在 z=0 平面上，作为全局参照
        try:
            grid_lines = []
            size = 400  # 半幅长度，栅格覆盖 [-size, size]
            step = 5   # 栅格间隔
            for x in range(-size, size + 1, step):
                grid_lines.append([[x, -size, 0], [x, size, 0]])
            for y in range(-size, size + 1, step):
                grid_lines.append([[-size, y, 0], [size, y, 0]])

            pts = []
            idx = []
            for i, (p1, p2) in enumerate(grid_lines):
                pts.append(p1)
                pts.append(p2)
                idx.append([2 * i, 2 * i + 1])

            self.global_grid = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts),
                lines=o3d.utility.Vector2iVector(idx)
            )
            self.global_grid.paint_uniform_color([0.7, 0.7, 0.7])
            self.vis.add_geometry(self.global_grid)
        except Exception:
            # 如果创建失败，也不影响主流程
            self.global_grid = None

        # 设置渲染选项
        opt = self.vis.get_render_option()
        opt.show_coordinate_frame = False
        opt.point_size = 2.0

        # 初始化固定的全局视角（远景），确保栅格和全局地图都在视野中
        try:
            view_ctl = self.vis.get_view_control()
            # 固定观察中心在原点（栅格中心），并将摄像机拉后、抬高
            view_ctl.set_lookat([0.0, 0.0, 0.0])
            view_ctl.set_front([0.0, -0.6, -0.8])
            view_ctl.set_up([0.0, 0.8, -0.6])
            # 缩放值选一个能看到整个栅格的合适初始值
            try:
                view_ctl.set_zoom(4.0)
            except Exception:
                # 某些Open3D版本对 zoom 限制严格，忽略异常
                pass
        except Exception:
            pass

    def update_o3d_visualization(self, frame_count: int):
        """更新Open3D可视化并捕获帧（改进：全局栅格+小相机框显示）"""
        if not self.vis:
            return
        
        self.vis_map = self.tsdf_volume.extract_point_cloud()

        # 1. 更新全局地图（只更新 points/colors），并避免让点云本身调整视角
        if len(self.vis_map.points) > 0:
            # 这里选择对 global_map 做一次下采样的副本用于可视化，避免可视化过重
            try:
                vis_map = self.vis_map.voxel_down_sample(max(0.0, self.voxel_size_map))
            except Exception:
                vis_map = self.vis_map

            self.map_geometry.points = vis_map.points
            if vis_map.has_colors():
                self.map_geometry.colors = vis_map.colors
            self.vis.update_geometry(self.map_geometry)

        # 2. 更新 pose（显示为小坐标系）——使用复制替换避免累计变换
        try:
            new_pose = copy.deepcopy(self.initial_pose_frame)
            new_pose.transform(self.global_transform)

            # 安全替换可视化中的 pose
            try:
                self.vis.remove_geometry(self.pose_frame, reset_bounding_box=False)
            except Exception:
                pass
            self.pose_frame = new_pose
            self.vis.add_geometry(self.pose_frame, reset_bounding_box=False)
            self._vis_has_pose = True
        except Exception:
            pass

        # 3. 更新轨迹（LineSet）——轨迹点在全局坐标系中连线
        current_center = self.global_transform[:3, 3].copy()
        if len(self.trajectory_points) == 0 or not np.allclose(self.trajectory_points[-1], current_center):
            self.trajectory_points.append(current_center)

        if len(self.trajectory_points) > 1:
            points_np = np.asarray(self.trajectory_points)
            lines_np = np.array([[i, i + 1] for i in range(len(points_np) - 1)])
            self.trajectory.points = o3d.utility.Vector3dVector(points_np)
            self.trajectory.lines = o3d.utility.Vector2iVector(lines_np)
            self.trajectory.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(lines_np), 1)))

            try:
                self.vis.remove_geometry(self.trajectory, reset_bounding_box=False)
            except Exception: 
                pass
            self.vis.add_geometry(self.trajectory, reset_bounding_box=False)
            self._vis_has_trajectory = True

        # 4. 保留全局栅格（不作修改）——如果栅格存在，确保它在渲染器里
        if hasattr(self, 'global_grid') and self.global_grid is not None:
            try:
                # 尝试确保栅格在可视化中（add/remove 可能重复，但无害）
                self.vis.update_geometry(self.global_grid)
            except Exception:
                pass

        # 5. 强制刷新渲染和事件处理（保持固定视角，因此不会因为地图改变跳动）
        for _ in range(3):
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)

        # 6. 捕获帧用于GIF
        frame_filename = os.path.join(self.temp_dir, f"frame_{frame_count:04d}.png")
        try:
            self.vis.capture_screen_image(frame_filename)
            self.frame_paths.append(frame_filename)
        except Exception:
            pass

    def build_map_interactive(self, output_file: str = "prior_map.ply", gif_output_file: str = "map_build_process.gif"):
        """交互式构建地图（主循环）"""
        if not self.start_camera():
            return

        self.temp_dir = tempfile.mkdtemp()
        self.setup_o3d_visualizer()

        print("开始构建点云地图(TSDF融合)...")
        print("移动相机以覆盖整个环境")
        print("按 'c' 捕获当前点云，按 's' 保存并退出，按 'q' 退出不保存")

        frame_count = 0
        is_initialized = False

        try:
            while True:
                
                data_pack = self.capture_frame_data()
                if not data_pack: continue
                current_pcd, rgbd, color_img, depth_img = data_pack

                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    # 捕获点云和ORB特征
                    pcd_high_res = current_pcd.voxel_down_sample(self.voxel_size_map)
                    pcd_icp_source = self.preprocess_pointcloud(current_pcd)
                    print(f"点云捕获完成")

                    if not is_initialized:
                        self.integrate_tsdf(rgbd, self.global_transform)
                        self.update_tracking_map(pcd_icp_source, self.global_transform)
                        # 提取 ORB 并添加关键帧 (携带 RGBD 对象)
                        pts, descs = self.extract_orb(color_img, depth_img)
                        self.add_keyframe(pcd_icp_source, pts, descs, rgbd)
                        
                        is_initialized = True
                        print("初始化完成")

                    else:
                         # --- Frame-to-Model 追踪 ---
                        # Target: 追踪地图 (World Coordinates)
                        result = self.robust_icp_registration(pcd_icp_source, self.tracking_map)

                        if result.fitness > 0.3:
                            # 更新位姿
                            self.global_transform = result.transformation
                            
                            # 融合 TSDF
                            self.integrate_tsdf(rgbd, self.global_transform)
                            
                            # 关键帧判定
                            if self.is_keyframe(self.global_transform):
                                # 1. 更新追踪地图
                                self.update_tracking_map(pcd_icp_source, self.global_transform)
                                
                                # 2. 提取 ORB 并添加关键帧 (携带 RGBD 对象)
                                pts, descs = self.extract_orb(color_img, depth_img)
                                self.add_keyframe(pcd_icp_source, pts, descs, rgbd)
                                
                                # 3. 回环检测 (只对关键帧进行)
                                if len(self.keyframes) > 20: 
                                    self._detect_and_correct_loop(self.keyframes[-1])
                        else:
                            print(f"追踪丢失 (Fitness: {result.fitness:.3f})") 
                    self.update_o3d_visualization(frame_count)
                    frame_count += 1

                    print(f"已捕获 {frame_count} 帧点云，总点数: {len(self.vis_map.points)}")

                elif key == ord('s'):
                    # 保存地图
                    # --- 最终处理：调用 PGO 和重融合 ---
                    self.optimize_global_map(output_file)
                    self.save_map(output_file)
                    self.generate_gif(gif_output_file)
                    break

                elif key == ord('q'):
                    print("退出地图构建")
                    break

                if self.vis and not self.vis.poll_events():
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            if self.vis:
                self.vis.destroy_window()
            
            # # --- 最终处理：调用 PGO 和重融合 ---
            # self.optimize_global_map(output_file)

            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"临时文件目录 {self.temp_dir} 已清理。")
    

    def generate_gif(self, filename: str):
        """将捕获的帧生成为GIF动画"""
        if not self.frame_paths:
            print("没有捕获任何帧，无法生成GIF。")
            return

        print(f"开始生成GIF动画，共 {len(self.frame_paths)} 帧...")
        try:
            images = [iio.imread(path) for path in self.frame_paths if os.path.exists(path)]
            if not images:
                print("没有有效的帧图像，跳过GIF生成。")
                return
            iio.imwrite(filename, images, duration=100, loop=0)
            print(f"GIF动画已保存到 {filename}")
        except Exception as e:
            print(f"生成GIF失败: {e}")

    def save_map(self, filename: str):
        """保存点云地图"""
        if len(self.vis_map.points) == 0:
            print("没有点云数据可保存")
            return

        # 预处理：去噪和下采样
        # self.global_map = self.global_map.voxel_down_sample(self.voxel_size_map)

        # 可视化点云
        o3d.visualization.draw_geometries([self.vis_map], window_name="最终点云地图")
        # 保存为PLY文件
        o3d.io.write_point_cloud(filename, self.vis_map)
        print(f"点云地图已保存到 {filename}, 总点数: {len(self.vis_map.points)}")

        # 2. 保存 ORB 特征 (.npz)
        orb_points_filename = filename.replace(".ply", "_orb_features.npz")

        try:
            save_dict = {}

            # ---------- 1) 兼容旧格式：继续保存全局点池 ----------
            if self.global_orb_points and self.global_orb_descriptors:
                points_array = np.array(self.global_orb_points, dtype=np.float32)   # [x, y, z, frame_id]
                descriptors_array = np.array(self.global_orb_descriptors, dtype=np.uint8)

                if descriptors_array.size > 0:
                    save_dict["points"] = points_array
                    save_dict["descriptors"] = descriptors_array
                    print(f"全局 ORB 点池准备保存: {len(points_array)} points")

            # ---------- 2) 新格式：保存关键帧数据库 ----------
            keyframes_serializable = []
            for kf in self.keyframes:
                pts3d = kf.get("points_3d", None)
                desc = kf.get("descriptors", None)
                pose = kf.get("pose", np.eye(4))

                # 过滤掉空关键帧
                if pts3d is None or desc is None:
                    continue
                if len(pts3d) == 0 or len(desc) == 0:
                    continue

                keyframes_serializable.append({
                    "id": int(kf.get("id", -1)),
                    "T_map_cam": np.asarray(pose, dtype=np.float32),
                    "points3d": np.asarray(pts3d, dtype=np.float32),
                    "descriptors": np.asarray(desc, dtype=np.uint8),
                })

            save_dict["keyframes"] = np.array(keyframes_serializable, dtype=object)

            if len(keyframes_serializable) == 0 and "points" not in save_dict:
                print("没有 ORB 特征数据可保存。")
            else:
                np.savez_compressed(orb_points_filename, **save_dict)
                print(f"ORB特征已保存到 {orb_points_filename}")
                print(f"关键帧数: {len(keyframes_serializable)}")

        except Exception as e:
            print(f"保存 ORB 特征失败: {e}")


# 构建地图的使用示例

def build_map_example():
    builder = PointCloudMapBuilder()
    # 地图保存路径
    map_path = "my_environment_map_pgo.ply"
    # GIF保存路径
    gif_path = "map_build_process_pgo.gif"
    builder.build_map_interactive(map_path, gif_path)


# 如果直接运行此文件，可以选择构建地图或运行定位
if __name__ == "__main__":
    build_map_example()

