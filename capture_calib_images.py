#!/usr/bin/env python3
"""使用与 main.py 相同的摄像头模式采集 Charuco 标定图片。

按键：
  s：保存当前帧到 charuco_images/
  q 或 ESC：退出
"""

from pathlib import Path
import time

import cv2


CAMERA_INDEX = 2
WIDTH = 640
HEIGHT = 480
FPS = 30
OUTPUT_DIR = Path("charuco_images")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 /dev/video{CAMERA_INDEX}")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(f"摄像头实际参数: {actual_width}x{actual_height}, "
          f"fps={actual_fps:.1f}, fourcc={fourcc}")

    if (actual_width, actual_height) != (WIDTH, HEIGHT):
        cap.release()
        raise RuntimeError(
            f"摄像头未使用要求的 {WIDTH}x{HEIGHT} 模式，实际为 "
            f"{actual_width}x{actual_height}；请检查摄像头支持的模式。"
        )

    existing = sorted(OUTPUT_DIR.glob("calib_*.png"))
    next_index = len(existing) + 1
    print("移动 Charuco 标定板覆盖不同位置和角度后拍摄。")
    print("按 s 保存，按 q 或 ESC 退出。")

    # 给摄像头自动曝光/白平衡一点稳定时间
    time.sleep(1.0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 无法读取摄像头帧")
                continue

            display = frame.copy()
            cv2.putText(display, "S: save   Q/ESC: quit", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Calibration Image Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                filename = OUTPUT_DIR / f"calib_{next_index:03d}.png"
                if cv2.imwrite(str(filename), frame):
                    print(f"已保存: {filename} ({frame.shape[1]}x{frame.shape[0]})")
                    next_index += 1
                else:
                    print(f"[WARN] 保存失败: {filename}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
