import cv2
import numpy as np
import os

# ================== 你修改后的正确参数 ==================
CHESSBOARD_SIZE = (9, 6)   # 你要求改成 9×6 内角点
SQUARE_SIZE = 0.017        # 17mm 不变
IMAGE_DIR = "calib_images"

# 3D 世界坐标（自动适配 9×6）
objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 这里也改成 9,6
objp *= SQUARE_SIZE

obj_points = []
left_points = []
right_points = []

# 检查文件夹
if not os.path.exists(IMAGE_DIR):
    print("❌ 错误：找不到 calib_images 文件夹")
    exit()

img_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")])
print(f"找到 {len(img_files)} 张图片")

if len(img_files) == 0:
    print("❌ 错误：没有图片！先运行 capture.py 拍照！")
    exit()

# 开始检测
for fname in img_files:
    img = cv2.imread(os.path.join(IMAGE_DIR, fname))
    if img is None: continue

    h, w = img.shape[:2]
    w_half = w // 2
    left_img = img[:, :w_half]
    right_img = img[:, w_half:]

    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 找棋盘格
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # ================== 绘制角点 + 显示 + 等待按键 ==================
    cv2.drawChessboardCorners(left_img, CHESSBOARD_SIZE, corners_l, ret_l)
    cv2.drawChessboardCorners(right_img, CHESSBOARD_SIZE, corners_r, ret_r)

    combined = np.hstack((cv2.resize(left_img, (640, 360)), cv2.resize(right_img, (640, 360))))
    cv2.imshow(f"检测中 - {fname}", combined)

    print(f"👉 当前图片：{fname}，按任意键继续...")
    cv2.waitKey(2)
    cv2.destroyWindow(f"检测中 - {fname}")
    # =================================================================

    if ret_l and ret_r:
        print(f"✅ 检测成功：{fname}")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)

        obj_points.append(objp)
        left_points.append(corners_l)
        right_points.append(corners_r)

        cv2.drawChessboardCorners(left_img, CHESSBOARD_SIZE, corners_l, ret_l)
        cv2.drawChessboardCorners(right_img, CHESSBOARD_SIZE, corners_r, ret_r)
        show = np.hstack((cv2.resize(left_img, (640, 360)), cv2.resize(right_img, (640, 360))))
        cv2.imshow("Check", show)
        cv2.waitKey(100)
    else:
        print(f"❌ 检测失败：{fname}")

cv2.destroyAllWindows()

# ================== 关键判断 ==================
if len(obj_points) == 0:
    print("\n❌ 没有任何一张图检测到棋盘格！标定无法继续！")
    print("请检查：")
    print("1. 棋盘格内角点是否为 9×6")
    print("2. 图片是否清晰、不反光、不变形？")
    print("3. 标定板是否完整出现在左右画面？")
    exit()

print(f"\n有效图片：{len(obj_points)} 张，开始标定...")

# 单目 + 双目标定
img_size = (1920, 1080)
_, mtx_l, dist_l, _, _ = cv2.calibrateCamera(obj_points, left_points, img_size, None, None)
_, mtx_r, dist_r, _, _ = cv2.calibrateCamera(obj_points, right_points, img_size, None, None)

ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    obj_points, left_points, right_points,
    mtx_l, dist_l, mtx_r, dist_r, img_size,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC
)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T)
np.savez("stereo_calib_result.npz", mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r, R=R, T=T, Q=Q)

print("\n🎉 标定完成！参数已保存！")