import cv2
import numpy as np
import os

# ================== 修正后的硬件参数（完全匹配你的相机）==================
CHESSBOARD_SIZE = (9, 6)  # 9×6内角点
SQUARE_SIZE = 0.017  # 17mm方格，用卡尺实际测量后修改！
IMAGE_DIR = "calib_images"
# 🔴 致命错误修复：你的相机是左右拼接3840×1080，单目1920×1080
IMG_SIZE = (3840, 1080)  # 必须与相机实际输出完全一致
SINGLE_IMG_SIZE = (1920, 1080)

# 3D世界坐标
objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []
left_points = []
right_points = []
valid_filenames = []

# 检查文件夹
if not os.path.exists(IMAGE_DIR):
    print("❌ 错误：找不到 calib_images 文件夹")
    exit()

img_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith((".png", ".jpg", ".bmp"))])
print(f"找到 {len(img_files)} 张图片")

if len(img_files) == 0:
    print("❌ 错误：没有图片！先运行 capture.py 拍照！")
    exit()

# 角点检测
print("\n开始检测棋盘格角点...")
for fname in img_files:
    img_path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图片：{fname}")
        continue

    h, w = img.shape[:2]
    if (w, h) != IMG_SIZE:
        print(f"❌ 错误：图片 {fname} 分辨率 {w}×{h} 与设置 {IMG_SIZE} 不符！")
        continue

    w_half = w // 2
    left_img = img[:, :w_half]
    right_img = img[:, w_half:]

    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 增强型角点检测
    ret_l, corners_l = cv2.findChessboardCorners(
        gray_l, CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FILTER_QUADS +
        cv2.CALIB_CB_FAST_CHECK
    )
    ret_r, corners_r = cv2.findChessboardCorners(
        gray_r, CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FILTER_QUADS +
        cv2.CALIB_CB_FAST_CHECK
    )

    # 实时显示
    cv2.drawChessboardCorners(left_img, CHESSBOARD_SIZE, corners_l, ret_l)
    cv2.drawChessboardCorners(right_img, CHESSBOARD_SIZE, corners_r, ret_r)
    combined = np.hstack((cv2.resize(left_img, (640, 360)), cv2.resize(right_img, (640, 360))))
    cv2.imshow("角点检测", combined)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        print(f"⏭️ 手动跳过：{fname}")
        continue

    if ret_l and ret_r:
        print(f"✅ 检测成功：{fname}")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        left_points.append(corners_l)
        right_points.append(corners_r)
        valid_filenames.append(fname)
    else:
        print(f"❌ 检测失败：{fname}")

cv2.destroyAllWindows()

if len(obj_points) < 20:
    print(f"\n❌ 有效图片不足20张（当前：{len(obj_points)}张），标定精度无法保证！")
    exit()

print(f"\n有效图片：{len(obj_points)} 张，开始单目标定...")

# 广角畸变模型
calib_flags = (
        cv2.CALIB_RATIONAL_MODEL +
        cv2.CALIB_FIX_K4 +
        cv2.CALIB_FIX_K5 +
        cv2.CALIB_FIX_K6
)

ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    obj_points, left_points, SINGLE_IMG_SIZE, None, None, flags=calib_flags
)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    obj_points, right_points, SINGLE_IMG_SIZE, None, None, flags=calib_flags
)

print(f"左相机单目标定重投影误差：{ret_l:.4f} 像素")
print(f"右相机单目标定重投影误差：{ret_r:.4f} 像素")

print("\n开始双目标定...")
stereo_ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    obj_points, left_points, right_points,
    mtx_l, dist_l, mtx_r, dist_r, SINGLE_IMG_SIZE,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
    flags=cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS
)

print(f"双目标定总重投影误差：{stereo_ret:.4f} 像素")
print(f"基线长度：{np.linalg.norm(T) * 1000:.2f} mm（理论值：61.00 mm）")
print(f"基线误差：{abs(np.linalg.norm(T) * 1000 - 61):.2f} mm")

# 立体校正
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, SINGLE_IMG_SIZE, R, T, alpha=0
)

# 逐图误差计算与坏图剔除
print("\n正在计算逐图重投影误差...")
total_error = 0
error_list = []
for i in range(len(obj_points)):
    imgpoints_l, _ = cv2.projectPoints(obj_points[i], rvecs_l[i], tvecs_l[i], mtx_l, dist_l)
    error_l = cv2.norm(left_points[i], imgpoints_l, cv2.NORM_L2) / len(imgpoints_l)

    imgpoints_r, _ = cv2.projectPoints(obj_points[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
    error_r = cv2.norm(right_points[i], imgpoints_r, cv2.NORM_L2) / len(imgpoints_r)

    avg_error = (error_l + error_r) / 2
    error_list.append(avg_error)
    total_error += avg_error
    print(f"图片 {valid_filenames[i]}：平均误差 {avg_error:.4f} 像素")

mean_error = total_error / len(obj_points)
print(f"\n平均重投影误差：{mean_error:.4f} 像素")

BAD_IMAGE_THRESHOLD = 0.15  # 1920×1080分辨率下的合格标准
bad_indices = [i for i, err in enumerate(error_list) if err > BAD_IMAGE_THRESHOLD]

if len(bad_indices) > 0:
    print(f"\n⚠️ 发现 {len(bad_indices)} 张误差过大的图片，将自动剔除：")
    for i in bad_indices:
        print(f"   - {valid_filenames[i]} (误差：{error_list[i]:.4f} 像素)")

    obj_points = [p for i, p in enumerate(obj_points) if i not in bad_indices]
    left_points = [p for i, p in enumerate(left_points) if i not in bad_indices]
    right_points = [p for i, p in enumerate(right_points) if i not in bad_indices]

    if len(obj_points) < 20:
        print(f"\n❌ 剔除坏图后有效图片不足20张（当前：{len(obj_points)}张），请重新采集！")
        exit()

    print("\n正在重新标定...")
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        obj_points, left_points, SINGLE_IMG_SIZE, None, None, flags=calib_flags
    )
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        obj_points, right_points, SINGLE_IMG_SIZE, None, None, flags=calib_flags
    )

    stereo_ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, left_points, right_points,
        mtx_l, dist_l, mtx_r, dist_r, SINGLE_IMG_SIZE,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
        flags=cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, SINGLE_IMG_SIZE, R, T, alpha=0
    )

    print(f"重新标定后平均重投影误差：{stereo_ret:.4f} 像素")
    print(f"重新标定后基线长度：{np.linalg.norm(T) * 1000:.2f} mm")

# 保存参数
np.savez(
    "stereo_calib_result.npz",
    mtx_l=mtx_l, dist_l=dist_l,
    mtx_r=mtx_r, dist_r=dist_r,
    R=R, T=T, Q=Q,
    R1=R1, R2=R2, P1=P1, P2=P2,
    roi1=roi1, roi2=roi2,
    img_size=SINGLE_IMG_SIZE
)

print("\n🎉 标定完成！所有参数已保存到 stereo_calib_result.npz")
print("\n📊 标定结果评估：")
print(f"   平均重投影误差：{stereo_ret:.4f} 像素")
print(f"   基线长度：{np.linalg.norm(T) * 1000:.2f} mm")
print(f"   基线误差：{abs(np.linalg.norm(T) * 1000 - 61):.2f} mm")

if stereo_ret < 0.08:
    print("✅ 标定结果优秀！可以用于高精度测量")
elif stereo_ret < 0.15:
    print("⚠️ 标定结果良好，可用于一般测量")
else:
    print("❌ 标定结果不合格！请重新采集图片并标定")
    print("   建议：增加近距离和中距离的图片数量")

# 极线校正验证
print("\n正在生成极线校正验证图...")
test_img = cv2.imread(os.path.join(IMAGE_DIR, valid_filenames[0]))
w_half = test_img.shape[1] // 2
left_test = test_img[:, :w_half]
right_test = test_img[:, w_half:]

map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, SINGLE_IMG_SIZE, cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, SINGLE_IMG_SIZE, cv2.CV_16SC2)

left_rect = cv2.remap(left_test, map1_l, map2_l, cv2.INTER_LINEAR)
right_rect = cv2.remap(right_test, map1_r, map2_r, cv2.INTER_LINEAR)


def draw_epipolar_lines(img1, img2, num_lines=20):
    h, w = img1.shape[:2]
    for i in range(num_lines):
        y = int(h * i / num_lines)
        cv2.line(img1, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(img2, (0, y), (w, y), (0, 255, 0), 1)
    return np.hstack((img1, img2))


epipolar_img = draw_epipolar_lines(left_rect, right_rect)
cv2.imwrite("epipolar_verification.png", epipolar_img)
cv2.imshow("极线校正验证", cv2.resize(epipolar_img, (1280, 360)))
print("极线校正验证图已保存为 epipolar_verification.png")
print("✅ 验证标准：左右图像的对应点应在同一条绿色水平线上")

cv2.waitKey(0)
cv2.destroyAllWindows()