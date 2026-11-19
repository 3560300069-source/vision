import cv2
import numpy as np
import os

# ================== 你的正确参数 ==================
CHESSBOARD_SIZE = (9, 6)
IMAGE_DIR = "calib_images"
CALIB_FILE = "stereo_calib_result.npz"

# 加载标定
data = np.load(CALIB_FILE)
mtx_l = data["mtx_l"]
dist_l = data["dist_l"]
mtx_r = data["mtx_r"]
dist_r = data["dist_r"]
R = data["R"]
T = data["T"]

w, h = 1920, 1080

# 极线校正
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w, h), cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w, h), cv2.CV_32FC1)

# ================== 遍历图片 ==================
total_error = 0.0
valid_count = 0

# 生成棋盘格3D点（用于计算误差）
objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp *= 0.017  # 你的方格大小17mm

for fname in os.listdir(IMAGE_DIR):
    img = cv2.imread(os.path.join(IMAGE_DIR, fname))
    if img is None: continue

    # ================== ✅ 相机正确，标准切分 ==================
    left_raw = img[:, :1920]   # 左半 = 左目
    right_raw = img[:, 1920:]  # 右半 = 右目
    # ==========================================================

    # 极线校正
    left_rect = cv2.remap(left_raw, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_raw, map1_r, map2_r, cv2.INTER_LINEAR)

    combined = np.hstack((left_rect, right_rect))
    scale = 0.5
    show_img = cv2.resize(combined, None, fx=scale, fy=scale)
    for y in range(0, int(h * scale), 20):
        cv2.line(show_img, (0, y), (int(3840 * scale), y), (0, 255, 0), 1)

    cv2.imshow("【正常显示】极线对齐  按任意键下一张", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 在原始未校正图计算误差
    gray_l = cv2.cvtColor(left_raw, cv2.COLOR_BGR2GRAY)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, None)

    if ret_l:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)

        img_proj, _ = cv2.projectPoints(objp, (0,0,0), (0,0,0), mtx_l, dist_l)
        error = cv2.norm(corners_l, img_proj, cv2.NORM_L2) / len(img_proj)
        total_error += error
        valid_count += 1

# ================== 最终输出 ==================
if valid_count > 0:
    avg_error = total_error / valid_count
    print("\n" + "="*50)
    print(f"✅ 有效图片数量：{valid_count}")
    print(f"✅ 平均重投影误差：{avg_error:.4f} 像素")
    print("="*50)

    if avg_error < 0.5:
        print("✅ 标定质量：优秀，可做高精度深度图！")
    elif avg_error < 1.0:
        print("✅ 标定质量：良好，正常使用无问题")
    else:
        print("⚠️ 标定质量一般，可优化")
else:
    print("❌ 未检测到有效棋盘格")