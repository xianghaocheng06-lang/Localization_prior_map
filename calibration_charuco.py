#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

# ====================== 配置 ======================
# 标定图片必须使用与 main.py 相同的 640x480 采集模式
ARUCO_DICT = cv2.aruco.DICT_4X4_50
SQUARES_X =11
SQUARES_Y = 8
SQUARE_LENGTH = 0.015   # 打印后实际测量！
MARKER_LENGTH = 0.011
IMAGE_DIR = "charuco_images"
EXPECTED_IMAGE_SIZE = (640, 480)  # 必须与 main.py 和拍照脚本一致
# =================================================

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
detector = cv2.aruco.CharucoDetector(board)   # 现代推荐方式

all_charuco_corners = []
all_charuco_ids = []
all_object_points = []
all_image_points = []
image_size = None

images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png")))
print(f"找到 {len(images)} 张图片")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    if (img.shape[1], img.shape[0]) != EXPECTED_IMAGE_SIZE:
        print(f"跳过分辨率不匹配的图片: {fname} ({img.shape[1]}x{img.shape[0]})")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = EXPECTED_IMAGE_SIZE

    # 检测 Charuco
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_ids is not None and len(charuco_ids) > 4:  # 至少需要几个点
        # 匹配 3D-2D 点
        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_pts is not None and len(obj_pts) > 0:
            all_object_points.append(obj_pts)
            all_image_points.append(img_pts)
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

            # 可视化
            cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
            cv2.imshow("Charuco", img)
            cv2.waitKey(100)
            print(f"有效: {fname}, 检测到 {len(charuco_ids)} 个点")

cv2.destroyAllWindows()

if len(all_object_points) < 10:
    print("有效图片不足！")
    exit()

# 标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    all_object_points, all_image_points, image_size, None, None
)

print(f"\n重投影误差: {ret:.4f}")
print("Camera Matrix:\n", camera_matrix)
print("Distortion:\n", dist_coeffs.ravel())

np.savez("charuco_calib.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rms=ret)
print("结果已保存")
