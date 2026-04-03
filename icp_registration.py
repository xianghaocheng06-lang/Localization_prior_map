import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import copy
from typing import Optional, Tuple

class ICPLocalization:

    def __init__(self):

        # --- 配置区域 ---
        self.enable_colored_icp = False  # 【重要】默认关闭 Colored ICP 以提升速度
        self.process_every_n_frames = 4  # 【重要】每隔几帧处理一次 (降频以减少延迟)
        self.show_pointcloud_realtime = False # 【重要】实时显示是否刷新点云？False则只显示坐标轴(最快)
        # ----------------

        # 相机参数
        self.camera_matrix = np.array([
            [615.0, 0, 320.0],
            [0, 615.0, 240.0],
            [0, 0, 1]
        ])

        #初始化Realsense管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        #点云对齐

        self.align = rs.align(rs.stream.color)
        self.pointcloud = rs.pointcloud()

        #先验地图点云
        self.prior_map_pcd: Optional[o3d.geometry.PointCloud] = None
        self.current_frame_pcd: Optional[o3d.geometry.PointCloud] = None

        # 定位状态
        self.is_localized = False
        self.current_pose = np.eye(4)
        self.last_pose = np.eye(4)
        self.velocity = np.eye(4)
        self.frame_counter = 0

        # ICP参数  
        self.voxel_size_icp = 0.003

         # ICP参数优化
        self.coarse_threshold = self.voxel_size_icp * 10
        self.fine_threshold = self.voxel_size_icp

        # 深度范围（D405 适用于近距离）
        self.depth_min = 0.07
        self.depth_max = 0.6 

        # 深度滤波器初始化
        self.setup_depth_filters()

        # 可视化相关对象
        self.vis: Optional[o3d.visualization.Visualizer] = None
        self.pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) # 相机姿态坐标轴
        self.trajectory_points = [] # 存储相机中心点的列表
        self.trajectory = o3d.geometry.LineSet() # 用于绘制轨迹

    def start_stream(self):
        """启动Realsense流"""
        try:
            self.pipeline.start(self.config)
            print("Realsense stream started.")
            return True
        except Exception as e:
            print(f"Failed to start Realsense stream: {e}")
            return False
    
    def load_prior_map(self, pcd_path: str):
        """加载先验地图点云"""
        try:
            self.prior_map_pcd = o3d.io.read_point_cloud(pcd_path)
            self.prior_map_pcd = self.preprocess_pointcloud(self.prior_map_pcd)
            # --- 新增：预计算地图的 FPFH 特征 ---
            print("正在预计算地图特征...")
            self.map_down, self.map_fpfh = self.preprocess_pointcloud_features(self.prior_map_pcd)

            # --------------------------------
            o3d.visualization.draw_geometries([self.prior_map_pcd], window_name="全局点云地图")
            print(f"Prior map loaded from {pcd_path}.")
            return True
        except Exception as e:
            print(f"加载先验地图失败: {e}")
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
    
    def preprocess_pointcloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """点云预处理"""
        # 下采样
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size_icp)
        
        # 估计法线（ICP需要法线信息）
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_icp * 2.0, max_nn=30))
        
        return pcd 
    
    def capture_frame(self):
        """捕获对齐后的深度和彩色帧"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Failed to capture frames.")
            return None, None
        
        # 深度滤波
        depth_frame = self._apply_depth_filters(depth_frame)

        # depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        return depth_frame, color_frame
    
    def depth_frame_to_pointcloud(self, depth_frame, color_frame) -> o3d.geometry.PointCloud:

        """将深度帧转换为Open3D点云"""
        points = self.pointcloud.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        valid_mask = ~np.isnan(vertices).any(axis=1)
        valid_mask &= (vertices[:, 2] > self.depth_min)
        valid_mask &= (vertices[:, 2] < self.depth_max)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices[valid_mask])

        color_image = np.asanyarray(color_frame.get_data())
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image_flat = color_image_rgb.reshape(-1, 3)
        colors = color_image_flat[valid_mask]
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        return pcd
    
    def preprocess_pointcloud_features(self, pcd):
        """为全局配准提取几何特征 (FPFH)"""
        # 使用较大的 voxel size 进行下采样，提取特征不需要太细
        voxel_size = self.voxel_size_icp * 5 
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    
    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        """使用 RANSAC 进行全局配准"""
        print("正在进行全局搜索 (RANSAC)...")
        
        # 1. 准备参数
        # 特征提取用的体素大小 (通常是 ICP voxel 的 5倍)
        voxel_size = self.voxel_size_icp * 5 
        
        # RANSAC 的距离阈值，通常设为 voxel_size 的 1.5 倍
        # 这比 FGR (0.5倍) 宽松，因为 RANSAC 的目的是找到“大概正确”的位置
        distance_threshold = voxel_size * 1.5 

        # 2. 执行 RANSAC
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, 
            target_down, 
            source_fpfh, 
            target_fpfh,
            mutual_filter=True,  # 开启相互滤波器，剔除单向匹配的异常点
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3, # 每次采样 3 个点来计算变换
            checkers=[
                # 几何约束：检查源点云中两点距离与目标点云中两点距离是否一致 (0.9 表示允许 10% 误差)
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                # 距离约束：检查对齐后的点距离是否小于阈值
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            # 迭代次数 100000，置信度 0.999
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        return result

    def icp_registration(self, source_pcd: o3d.geometry.PointCloud, 
                        target_pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, float]:
        """执行ICP配准"""
        ###### 恒定速度模型 ########
        try:
            predicted_pose = self.current_pose @ self.velocity
        except:
            predicted_pose = self.current_pose


        try:
            # 预处理当前点云
            source_pcd = self.preprocess_pointcloud(source_pcd)
            
             # 2. 粗配准
            result_icp_coarse = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, self.coarse_threshold, predicted_pose,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            
            # 3. 精配准
            result_icp_fine = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, self.fine_threshold, result_icp_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )

            if self.enable_colored_icp:
                try:
                    # Colored ICP
                    result_icp_fine = o3d.pipelines.registration.registration_colored_icp(
                        source_pcd, target_pcd, self.voxel_size_icp, result_icp_fine.transformation,
                        o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.5), # 降低几何权重，更信赖颜色
                        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30)
                    )
                except Exception as e:
                    pass

            
            # 返回变换矩阵和配准得分
            return result_icp_fine.transformation, result_icp_fine.fitness
            
        except Exception as e:
            print(f"ICP配准失败: {e}")
            return np.eye(4), 0.0
        
    def localize_with_icp(self, depth_frame, color_frame):
        """使用ICP进行定位"""
        if self.prior_map_pcd is None:
            print("错误: 未加载先验地图")
            return False
        
        # 转换为点云
        current_pcd = self.depth_frame_to_pointcloud(depth_frame, color_frame)
        
        if len(current_pcd.points) < 100:  # 点云太稀疏
            print("当前点云太稀疏，跳过配准")
            return False
        
        if self.frame_counter == 1:
            source_down, source_fpfh = self.preprocess_pointcloud_features(current_pcd)
            result_global = self.execute_global_registration(
                source_down, self.map_down, source_fpfh, self.map_fpfh
            )
            if result_global.fitness > 0.4 and result_global.inlier_rmse < 0.01: # RANSAC 的 fitness 阈值通常比 ICP 低
                print(f"RANSAC全局重定位成功! Fitness: {result_global.fitness}")
                self.current_pose = result_global.transformation
                # 用 RANSAC 的结果作为 ICP 的初值再精修一下
                transformation, fitness = self.icp_registration(current_pcd, self.prior_map_pcd)
                self.current_pose = transformation
                self.is_localized = True
                self.current_frame_pcd = current_pcd
                return True
            else:
                print("RANSAC全局重定位失败，请移动相机尝试更多特征区域...")
                return False
        else:
            # 执行ICP配准
            transformation, fitness = self.icp_registration(current_pcd, self.prior_map_pcd)

            
            # 检查配准质量
            if fitness > 0.3:  # 配准质量阈值
                self.velocity = transformation @ np.linalg.inv(self.current_pose)
                self.last_pose = self.current_pose
                self.current_pose = transformation
                self.current_frame_pcd = current_pcd
                self.is_localized = True
                print(f"ICP定位成功! 配准得分: {fitness:.3f}")
                return True
            else:
                print(f"ICP定位丢失! Fitness: {fitness:.3f}.")
                self.velocity = np.eye(4)
                self.is_localized = False
                return False
        
    def update_visualization(self):
        """更新Open3D可视化器中的几何体"""
        if not self.vis:
            return

        # 1. 更新相机姿态坐标系 (Pose Frame)
        # 相机姿态 T_map_camera (从地图到相机的变换)
        T_map_camera = self.current_pose 
        
        # 姿态坐标系的位置就是 T_map_camera 的平移部分
        current_center = T_map_camera[:3, 3] 
        
        # 更新坐标系几何体
        # self.pose_frame.transform(T_map_camera @ np.linalg.inv(self.last_center)) 
        # 重置到原点，再应用新的变换 (更安全的做法是直接创建一个新的，但为了实时性，我们尝试变换)
        
        # 姿态更新：Open3D 没有直接的 set_transform，使用 transform 更新。
        # NOTE: 保持姿态坐标系大小不变
        self.pose_frame.transform(np.linalg.inv(self.last_pose)) # 移除旧的变换
        self.pose_frame.transform(T_map_camera) # 应用新的变换
        
        self.last_pose = T_map_camera # 存储当前的 T_map_camera
        
        
        # 2. 更新当前帧点云 (Current Frame PCD)
        # 清空旧的点云数据，并加载新的点云
        # self.current_frame_pcd.points = self.current_frame_pcd.points
        # if self.current_frame_pcd and self.current_frame_pcd.points:
        #     # 为当前帧上色（绿色）
        #     self.current_frame_pcd.paint_uniform_color([0, 1, 0]) 
            
        #     # 将当前帧变换到全局坐标系
        #     current_frame_pcd_transformed = copy.deepcopy(self.current_frame_pcd)
        #     current_frame_pcd_transformed.transform(self.current_pose)

        #     # 更新几何体
        self.vis.update_geometry(self.pose_frame)
        #     self.vis.update_geometry(self.current_frame_pcd)

        # 3. 更新轨迹 (Trajectory)
        if len(self.trajectory_points) == 0 or not np.allclose(self.trajectory_points[-1], current_center):
            self.trajectory_points.append(current_center)

        if len(self.trajectory_points) > 1:
            points_np = np.asarray(self.trajectory_points)
            lines_np = np.array([[i, i + 1] for i in range(len(points_np) - 1)])

            self.trajectory.points = o3d.utility.Vector3dVector(points_np)
            self.trajectory.lines = o3d.utility.Vector2iVector(lines_np)
            self.trajectory.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(lines_np), 1))) # 蓝色轨迹
            self.vis.update_geometry(self.trajectory)

        # 3. 调整相机视角以跟随姿态 (解决超出视野的问题)
        # 将可视化器的焦点设置到当前相机位置，实现“跟随”效果
        # view_ctl = self.vis.get_view_control()
        # view_ctl.set_lookat(current_center) 

        # 4. 刷新可视化
        self.vis.poll_events()
        self.vis.update_renderer()
        

    def get_position(self) -> Optional[np.ndarray]:
        if self.is_localized:
            return self.current_pose[:3, 3]
        return None
    
    def get_rotation(self) -> Optional[np.ndarray]:
        if self.is_localized:
            return self.current_pose[:3, :3]
        return None
    
    def vis_pcd(self):

        if self.prior_map_pcd is None or self.current_frame_pcd is None:
            return
        
        prior_map_pcd_copy = self.prior_map_pcd.clone()
        current_frame_pcd_copy = self.current_frame_pcd.clone()

        prior_map_pcd_copy.paint_uniform_color([1, 0, 0]) # 红色表示先验地图
        current_frame_pcd_copy.paint_uniform_color([0, 1, 0]) # 绿色表示当前帧

        current_frame_pcd_copy.transform(self.current_pose)

        o3d.visualization.draw_geometries([prior_map_pcd_copy, current_frame_pcd_copy])
    
    def run_localization(self):
        """主定位循环"""

        # 启动可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="ICP 实时定位可视化", width=1024, height=768)

        # 加载地图和初始几何体到可视化器
        if not self.prior_map_pcd.has_colors():
             self.prior_map_pcd.paint_uniform_color([0.7, 0.7, 0.7])
        # self.prior_map_pcd.paint_uniform_color([0.7, 0.7, 0.7]) # 红色表示先验地图
        self.vis.add_geometry(self.prior_map_pcd)
        self.vis.add_geometry(self.pose_frame)

        # 初始化当前帧点云对象（必须先添加几何体才能在循环中更新）
        # self.current_frame_pcd = o3d.geometry.PointCloud()
        # self.current_frame_pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3))) # 初始化一个点
        # self.vis.add_geometry(self.current_frame_pcd)
        self.vis.add_geometry(self.trajectory)

        # 调整初始相机视角
        view_ctl = self.vis.get_view_control()
        map_center = self.prior_map_pcd.get_center()
        view_ctl.set_lookat(map_center) 
        view_ctl.set_up([0, -1, 0])      
        view_ctl.set_front([0, 0, -1])  
        view_ctl.set_zoom(0.5)

        localization_counter = 0
        try:
            while True:
                self.frame_counter += 1
                # 捕获深度帧
                depth_frame, color_frame = self.capture_frame()
                if self.frame_counter!=1 and self.frame_counter % self.process_every_n_frames != 0:
                    cv2.waitKey(1)
                    continue
                # if self.frame_counter == 1:
                #     current_pcd = self.depth_frame_to_pointcloud(depth_frame, color_frame)
                #     source_down, source_fpfh = self.preprocess_pointcloud_features(current_pcd)
                #     result_global = self.execute_global_registration(
                # source_down, self.map_down, source_fpfh, self.map_fpfh)
                #     # print(f"result_global.fitness: {result_global.fitness}")
                #     # if result_global.fitness > 0.05: 
                #     print(f"全局初始匹配成功! Fitness: {result_global.fitness:.3f}")
                #     self.current_pose = result_global.transformation
                #     final_transform, fitness = self.icp_registration(current_pcd, self.prior_map_pcd)
                #     if fitness > 0.4: # 如果 ICP 确认这个位置是对的
                #         self.current_pose = final_transform
                #         self.last_pose = final_transform
                #         self.is_localized = True
                #         pos = self.get_position()
                #         rot = self.get_rotation()
                #         print(f"位置: {pos}, 旋转矩阵:\n{rot}, 切换到ICP模式")
                #     else:
                #         print("全局匹配尚可，但 ICP 精修失败 (可能是伪匹配)。")
                success = self.localize_with_icp(depth_frame, color_frame)
                if success:
                    localization_counter += 1
                    pos = self.get_position()
                    rot = self.get_rotation()
                    print(f"定位次数: {localization_counter}, 位置: {pos}, 旋转矩阵:\n{rot}", )

                # 更新可视化
                self.update_visualization()

                # 检查可视化窗口是否关闭 (必须在循环中调用 poll_events)
                if not self.vis.poll_events():
                    break
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self.vis_pcd()
        except KeyboardInterrupt:
            print("定位中断")
        finally:
            self.pipeline.stop()
            self.vis.destroy_window()
            cv2.destroyAllWindows()
        
def main():
    """测试主函数"""
    localizer = ICPLocalization()

    if not localizer.start_stream():
        return
    
    prior_map_path = "my_environment_map_pgo.ply"
    if not localizer.load_prior_map(prior_map_path):
        return
    
    localizer.run_localization()

if __name__ == "__main__":
    main()
    # pcd_path  = "Code/ICP/wingbox_v6.ply"
    # prior_map_pcd = o3d.io.read_point_cloud(pcd_path)
    # o3d.visualization.draw_geometries([prior_map_pcd], window_name="全局点云地图")

                
    


