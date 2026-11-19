import cv2
import os

# 保存文件夹
SAVE_DIR = "test_images"
count = 0
def init_camera():
    cap = cv2.VideoCapture(0)
os.makedirs(SAVE_DIR, exist_ok=True)
count = 0

# 初始化相机（你唯一正确的模式）
def init_camera():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap

# 抓拍一帧
def grab_frame(cap):
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    return frame if ret else None

# ================== 主程序 ==================
cap = init_camera()
current_frame = None

cv2.namedWindow("Stereo Camera", cv2.WINDOW_NORMAL)

print("✅ 空格 = 抓拍")
print("✅ S = 保存整张 3840×1080")
print("✅ Q = 退出")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # 空格抓拍
    if key == ord(' '):
        current_frame = grab_frame(cap)
        print("✅ 已抓拍")

    # S保存整张图
    if key == ord('s') and current_frame is not None:
        save_path = os.path.join(SAVE_DIR, f"stereo_{count:02d}.png")
        cv2.imwrite(save_path, current_frame)
        print(f"✅ 已保存：{save_path}")
        count += 1

    # 显示
    if current_frame is not None:
        show = cv2.resize(current_frame, (0, 0), fx=0.4, fy=0.4)
        cv2.imshow("Stereo Camera", show)

cap.release()
cv2.destroyAllWindows()