import cv2
import numpy as np
import open3d as o3d

# ===================== 配置参数 =====================
IMG_PATH = r'C:\Users\ahuon\Desktop\IMG_0010.jpg'
# IMG_PATH = r'F:\Automatic-Number-Plate-Detection-main\Automatic-Number-Plate-Detection-main\vision3d\calib_images\stereo_28.png'
CALIB_FILE = "stereo_calib_result.npz"
W, H = 1920, 1080

# ✅ 修正后的视差范围（完美匹配你的61.4mm基线+2070焦距）
MIN_DISPARITY = 20       # 对应最远：2070*61.4/20 ≈ 6.35米
NUM_DISPARITIES = 416    # 必须是16的倍数，总范围20~436像素
                         # 对应最近：2070*61.4/436 ≈ 0.29米
BLOCK_SIZE = 11
SGBM_P1 = int(8 * 3 * BLOCK_SIZE ** 2)
SGBM_P2 = int(32 * 3 * BLOCK_SIZE ** 2)
DISP12_MAX_DIFF = 3
UNIQUENESS_RATIO = 12
SPECKLE_WINDOW_SIZE = 200
SPECKLE_RANGE = 40
# ====================================================

# 1. 加载标定参数
print("="*60)
print("🔍 标定参数确认:")
data = np.load(CALIB_FILE)
mtx_l = data["mtx_l"]
dist_l = data["dist_l"]
mtx_r = data["mtx_r"]
dist_r = data["dist_r"]
R = data["R"]
T = data["T"]

focal_length = mtx_l[0,0]
baseline = abs(T[0][0])
print(f"焦距 fx: {focal_length:.2f} 像素")
print(f"基线 B: {baseline*1000:.1f} 毫米")
print(f"理论检测范围: {focal_length*baseline/(MIN_DISPARITY+NUM_DISPARITIES):.2f}米 ~ {focal_length*baseline/MIN_DISPARITY:.2f}米")
print("="*60)

# 2. 极线校正
print("正在进行极线校正...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, (W, H), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (W, H), cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (W, H), cv2.CV_32FC1)

# 3. 读取并校正图像（修复JPEG读取问题）
print("正在读取图像...")
# 使用IMREAD_UNCHANGED避免OpenCV自动修复损坏的JPEG
full_img = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)
if full_img is None:
    print("❌ 图片读取失败！")
    print("解决方案：用画图软件打开图片，另存为新的PNG文件")
    exit()

left_raw = full_img[:, :W]
right_raw = full_img[:, W:]

left_rect = cv2.remap(left_raw, map1_l, map2_l, cv2.INTER_LINEAR)
right_rect = cv2.remap(right_raw, map1_r, map2_r, cv2.INTER_LINEAR)

left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

# 4. 计算视差图（纯SGBM，无需contrib模块）
print("正在计算视差图...")
sgbm = cv2.StereoSGBM_create(
    minDisparity=MIN_DISPARITY,
    numDisparities=NUM_DISPARITIES,
    blockSize=BLOCK_SIZE,
    P1=SGBM_P1,
    P2=SGBM_P2,
    disp12MaxDiff=DISP12_MAX_DIFF,
    uniquenessRatio=UNIQUENESS_RATIO,
    speckleWindowSize=SPECKLE_WINDOW_SIZE,
    speckleRange=SPECKLE_RANGE,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0

# 视差统计
valid_disp = disparity[(disparity > MIN_DISPARITY) & (disparity < MIN_DISPARITY + NUM_DISPARITIES)]
print(f"\n视差统计:")
print(f"  最小视差: {np.min(valid_disp):.2f}")
print(f"  最大视差: {np.max(valid_disp):.2f}")
print(f"  平均视差: {np.mean(valid_disp):.2f}")
print("="*60)

# 显示视差图
disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
cv2.imshow("视差图", cv2.resize(disp_color, (960, 540)))
cv2.waitKey(2000)

# 5. 生成3D点云
print("正在生成3D点云...")
points_3d = cv2.reprojectImageTo3D(disparity, Q)

mask = (disparity > MIN_DISPARITY) & (disparity < MIN_DISPARITY + NUM_DISPARITIES)
valid_points = points_3d[mask]
valid_colors = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)[mask] / 255.0

# 过滤异常点
min_distance = 0.3  # 0.3米
max_distance = 5.0  # 5米
distance_mask = (np.linalg.norm(valid_points, axis=1) < max_distance) & \
                (np.linalg.norm(valid_points, axis=1) > min_distance)
valid_points = valid_points[distance_mask]
valid_colors = valid_colors[distance_mask]

print(f"生成有效点: {len(valid_points):,} 个")
print(f"实际深度范围: {np.min(valid_points[:,2]):.2f} ~ {np.max(valid_points[:,2]):.2f} 米")
print("="*60)

# 6. 可视化
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)
pcd.colors = o3d.utility.Vector3dVector(valid_colors)

# 可选：点云下采样，提升流畅度
# pcd = pcd.voxel_down_sample(voxel_size=0.002)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="优化后的3D点云", width=1200, height=800)
vis.add_geometry(pcd)

opt = vis.get_render_option()
opt.background_color = np.array([0, 0, 0])
opt.point_size = 1.5
opt.show_coordinate_frame = True

vis.run()
vis.destroy_window()
cv2.destroyAllWindows()