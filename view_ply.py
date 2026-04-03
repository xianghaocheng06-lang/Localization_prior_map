import sys
import os
import open3d as o3d
import numpy as np

def create_scale_meter(size=1.0):
    """
    创建一个简单的 1单位 x 1单位 的线框立方体作为比例尺参考
    """
    # 创建一个对角线为 size 的立方体线框
    scale_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, 0, 0), max_bound=(size, size, size))
    )
    # 设置比例尺颜色为青色 (Cyan)
    scale_box.paint_uniform_color([0, 1, 1]) 
    return scale_box

def main():
    if len(sys.argv) < 2:
        print("用法: python view_ply.py <pointcloud.ply>")
        sys.exit(1)

    ply_path = sys.argv[1]

    if not os.path.exists(ply_path):
        print(f"文件不存在: {ply_path}")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(ply_path)

    if pcd.is_empty():
        print("点云为空，或者文件读取失败。")
        sys.exit(1)

    print("点云加载成功")
    print(f"点数: {len(pcd.points)}")

    # 获取点云的边界，用于自动调整坐标轴和比例尺的大小
    bbox = pcd.get_axis_aligned_bounding_box()
    max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())
    # 动态设定比例尺大小，设为最大跨度的 10% 
    scale_size = round(max_extent * 0.1, 1)
    print(f"参考比例尺单位长度: {scale_size}")

    # 1. 坐标轴 (红X, 绿Y, 蓝Z)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale_size, origin=[0, 0, 0])

    # 2. 比例尺线框 (青色方框，表示 scale_size^3 的体积)
    scale_ruler = create_scale_meter(scale_size)

    # 3. 可选：添加水平地面网格 (增加空间感)
    # 如果不需要可以从下面列表中移除
    
    print("\n操作提示:")
    print(" - 按 [H] 显示帮助")
    print(" - 按 [L] 切换光照")
    print(f" - 青色方框边长为: {scale_size} 单位")

    # 显示
    o3d.visualization.draw_geometries(
        [pcd, axes, scale_ruler],
        window_name=f"Open3D Viewer - Cyan Box = {scale_size} units",
        width=1280,
        height=720,
        point_show_normal=False
    )

if __name__ == "__main__":
    main()
