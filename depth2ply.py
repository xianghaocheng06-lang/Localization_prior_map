import open3d as o3d
import sys
import os
import numpy as np

def convert_depth_to_ply(depth_path):
    # 1. 检查文件是否存在
    if not os.path.exists(depth_path):
        print(f"错误: 找不到文件 {depth_path}")
        return

    # 2. 读取深度图
    depth_raw = o3d.io.read_image(depth_path)
    
    # 将 open3d 图像转为 numpy 数组以获取分辨率
    depth_array = np.asarray(depth_raw)
    height, width = depth_array.shape[:2]
    print(f"检测到图像分辨率: {width}x{height}")

    # 3. 自动计算 RealSense D405 相机内参
    # D405 的视场角 (FOV) 约为 87° x 58°
    # 焦距计算公式: f = width / (2 * tan(FOV_H / 2))
    # 对于 D405，一个非常准的近似经验值是: fx = fy = width * 0.5 左右（具体取决于具体的FOV设置）
    # 这里我们采用比例缩放法，以 1280x720 为基准
    scale = width / 1280.0
    
    fx = 640.0 * scale
    fy = 640.0 * scale
    cx = width / 2.0
    cy = height / 2.0

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )

    # 4. 从深度图转换成点云
    # RealSense D405 默认深度单位通常是 1mm = 1 unit，所以 scale 为 1000
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_raw, 
        intrinsic,
        depth_scale=1000.0, 
        depth_trunc=3.0, # 截断3米以外的无效点
        convert_rgb_to_intensity=False
    )

    # 5. 坐标系转换 (RealSense 坐标系 -> Open3D 标准坐标系)
    # 翻转 Y 和 Z 轴使点云正向显示
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 6. 保存点云
    output_path = os.path.splitext(depth_path)[0] + ".ply"
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"成功转换! 点云文件已保存至: {output_path}")

    # 7. 可视化
    print("正在打开可视化窗口...")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python depth_to_ply_auto.py <depth_image_path>")
    else:
        convert_depth_to_ply(sys.argv[1])
