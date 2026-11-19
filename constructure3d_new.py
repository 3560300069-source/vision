import cv2
import numpy as np
import open3d as o3d

# ===================== 配置参数（完全保留你的原结构）=====================
IMG_PATH = r'F:\Automatic-Number-Plate-Detection-main\Automatic-Number-Plate-Detection-main\vision3d\test_images\stereo_01.png'
# IMG_PATH = r'F:\Automatic-Number-Plate-Detection-main\Automatic-Number-Plate-Detection-main\vision3d\calib_images\stereo_17.png'

CALIB_FILE = "stereo_calib_result.npz"
W, H = 1920, 1080

# 视差参数（已针对你的相机优化，可微调）
MIN_DISPARITY = 16
NUM_DISPARITIES = 384  # 必须是16的倍数
BLOCK_SIZE = 9
SGBM_P1 = int(8 * 3 * BLOCK_SIZE ** 2)
SGBM_P2 = int(32 * 3 * BLOCK_SIZE ** 2)
DISP12_MAX_DIFF = 1
UNIQUENESS_RATIO = 10
SPECKLE_WINDOW_SIZE = 150
SPECKLE_RANGE = 2

# 🔴 新的滤波参数（大幅降低平滑强度）
SMOOTH_LEVEL = 1  # 平滑等级：0=几乎不滤波，1=轻度平滑，2=中度平滑，3=较强平滑
EDGE_THRESHOLD = 3  # 边缘保留阈值：1=最严格保留边缘，5=允许轻微模糊边缘
LR_CONSISTENCY_THRESHOLD = 1.5  # 左右一致性阈值：越大保留越多视差，越小越严格

# 点云后处理参数（也降低了去噪强度）
VOXEL_SIZE = 0.0015  # 从2mm改为1.5mm，保留更多细节
STATISTICAL_NEIGHBORS = 20  # 从30改为20，减少点云剔除
STATISTICAL_STD_RATIO = 1.5  # 从1.2改为1.5，放宽离群点标准
RADIUS_OUTLIER_NEIGHBORS = 6  # 从10改为6
RADIUS_OUTLIER_RADIUS = 0.007  # 从5mm改为7mm
# ====================================================

# 1. 加载标定参数（完全不变）
print("="*60)
print("🔍 标定参数确认:")
data = np.load(CALIB_FILE)
mtx_l = data["mtx_l"]
dist_l = data["dist_l"]
mtx_r = data["mtx_r"]
dist_r = data["dist_r"]
R = data["R"]
T = data["T"]
R1 = data["R1"]
R2 = data["R2"]
P1 = data["P1"]
P2 = data["P2"]
Q = data["Q"]

focal_length = mtx_l[0,0]
baseline = abs(T[0][0])
print(f"焦距 fx: {focal_length:.2f} 像素")
print(f"基线 B: {baseline*1000:.1f} 毫米")
print(f"理论检测范围: {focal_length*baseline/(MIN_DISPARITY+NUM_DISPARITIES):.2f}米 ~ {focal_length*baseline/MIN_DISPARITY:.2f}米")
print("="*60)

# 2. 极线校正（完全不变）
print("正在进行极线校正...")
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (W, H), cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (W, H), cv2.CV_16SC2)

# 3. 读取并校正图像（完全不变）
print("正在读取图像...")
full_img = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)
if full_img is None:
    print("❌ 图片读取失败！")
    exit()

left_raw = full_img[:, :W]
right_raw = full_img[:, W:]

left_rect = cv2.remap(left_raw, map1_l, map2_l, cv2.INTER_LINEAR)
right_rect = cv2.remap(right_raw, map1_r, map2_r, cv2.INTER_LINEAR)

left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

# 4. 计算视差图（大幅降低滤波强度）
print("正在计算视差图...")
# 初始化SGBM匹配器（完全不变）
left_matcher = cv2.StereoSGBM_create(
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

# 计算原始视差
disp_left = left_matcher.compute(left_gray, right_gray)
disp_left = disp_left.astype(np.float32) / 16.0

# 🔴 优化后的轻量级滤波流水线（保留细节优先）
print("正在进行轻量级视差滤波...")
filtered_disp = disp_left.copy()

# 步骤1：左右一致性检查（放宽阈值，保留更多视差）
right_matcher = cv2.StereoSGBM_create(
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
disp_right = right_matcher.compute(right_gray, left_gray)
disp_right = disp_right.astype(np.float32) / 16.0

# 左右一致性掩码（放宽阈值）
h, w = disp_left.shape
x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
x_right = x_coords - disp_left
x_right = x_right.astype(np.int32)

valid_mask = (x_right >= 0) & (x_right < w)
disp_left_valid = disp_left[valid_mask]
disp_right_valid = disp_right[y_coords[valid_mask], x_right[valid_mask]]

lr_mask = np.abs(disp_left_valid - disp_right_valid) < LR_CONSISTENCY_THRESHOLD
full_lr_mask = np.zeros_like(disp_left, dtype=bool)
full_lr_mask[valid_mask] = lr_mask

# 步骤2：可选的中值滤波（仅在SMOOTH_LEVEL≥2时开启）
if SMOOTH_LEVEL >= 2:
    filtered_disp = cv2.medianBlur(filtered_disp, 3)

# 步骤3：极轻量引导滤波（大幅降低平滑强度）
if SMOOTH_LEVEL >= 1:
    # 更小的滤波半径和更大的epsilon，保留更多细节
    guided_radius = 3 * EDGE_THRESHOLD
    guided_epsilon = 0.5 * EDGE_THRESHOLD
    # 使用OpenCV内置的双边滤波替代引导滤波（完全不依赖ximgproc）
    filtered_disp = cv2.bilateralFilter(
        filtered_disp,
        d=guided_radius,
        sigmaColor=guided_epsilon*10,
        sigmaSpace=guided_epsilon*10
    )

# 步骤4：应用左右一致性掩码（只剔除明显错误的视差）
filtered_disp[~full_lr_mask] = disp_left[~full_lr_mask]  # 不再直接置0，保留原始值

# 步骤5：极轻量形态学操作（仅在SMOOTH_LEVEL≥3时开启）
if SMOOTH_LEVEL >= 3:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    filtered_disp = cv2.morphologyEx(filtered_disp, cv2.MORPH_CLOSE, kernel)

# 步骤6：仅填充极小的孔洞
mask = (filtered_disp <= MIN_DISPARITY) | (filtered_disp >= MIN_DISPARITY + NUM_DISPARITIES)
if np.any(mask) and SMOOTH_LEVEL >= 1:
    # 只填充小于5x5的孔洞
    kernel = np.ones((3,3), np.uint8)
    small_holes = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) & ~mask
    filtered_disp[small_holes] = cv2.inpaint(filtered_disp, small_holes.astype(np.uint8), 2, cv2.INPAINT_TELEA)[small_holes]

# 视差统计（完全不变）
valid_disp = filtered_disp[(filtered_disp > MIN_DISPARITY) & (filtered_disp < MIN_DISPARITY + NUM_DISPARITIES)]
print(f"\n轻量级滤波后视差统计:")
print(f"  最小视差: {np.min(valid_disp):.2f}")
print(f"  最大视差: {np.max(valid_disp):.2f}")
print(f"  平均视差: {np.mean(valid_disp):.2f}")
print(f"  视差标准差: {np.std(valid_disp):.2f}")
print(f"  有效视差占比: {len(valid_disp)/(w*h)*100:.1f}%")
print("="*60)

# 显示对比图（完全不变）
disp_raw_normalized = cv2.normalize(disp_left, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
disp_filtered_normalized = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
disp_raw_color = cv2.applyColorMap(disp_raw_normalized, cv2.COLORMAP_JET)
disp_filtered_color = cv2.applyColorMap(disp_filtered_normalized, cv2.COLORMAP_JET)

comparison = np.hstack((disp_raw_color, disp_filtered_color))
cv2.putText(comparison, "原始视差", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
cv2.putText(comparison, "轻量级滤波后视差", (W+50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
cv2.imshow("视差对比图", cv2.resize(comparison, (1920, 540)))
cv2.waitKey(3000)

# 5. 生成3D点云（降低去噪强度）
print("正在生成3D点云...")
points_3d = cv2.reprojectImageTo3D(filtered_disp, Q)

mask = (filtered_disp > MIN_DISPARITY) & (filtered_disp < MIN_DISPARITY + NUM_DISPARITIES)
valid_points = points_3d[mask]
valid_colors = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)[mask] / 255.0

# 深度范围过滤（完全不变）
min_distance = 0.3
max_distance = 5.0
distance_mask = (np.linalg.norm(valid_points, axis=1) < max_distance) & \
                (np.linalg.norm(valid_points, axis=1) > min_distance)
valid_points = valid_points[distance_mask]
valid_colors = valid_colors[distance_mask]

# 轻量级点云去噪（大幅降低强度）
print("正在进行轻量级点云去噪...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)
pcd.colors = o3d.utility.Vector3dVector(valid_colors)

# 1. 更细的体素下采样
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

# 2. 更宽松的统计滤波
if SMOOTH_LEVEL >= 1:
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=STATISTICAL_NEIGHBORS,
        std_ratio=STATISTICAL_STD_RATIO
    )

# 3. 可选的半径滤波（仅在SMOOTH_LEVEL≥2时开启）
if SMOOTH_LEVEL >= 2:
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=RADIUS_OUTLIER_NEIGHBORS,
        radius=RADIUS_OUTLIER_RADIUS
    )

print(f"最终有效点: {len(pcd.points):,} 个")
print(f"实际深度范围: {np.min(np.asarray(pcd.points)[:,2]):.2f} ~ {np.max(np.asarray(pcd.points)[:,2]):.2f} 米")
print("="*60)

# 7. 可视化（完全不变）
print("正在打开可视化窗口...")
print("操作说明：左键旋转，右键平移，滚轮缩放，Q键退出")

# 解决GLFW上下文错误
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="保留细节的3D点云", width=1200, height=800, visible=True)

# 设置渲染选项
opt = vis.get_render_option()
opt.background_color = np.array([0, 0, 0])
opt.point_size = 1.5  # 从2.0改为1.5，更清晰显示细节
opt.show_coordinate_frame = True
opt.light_on = True

vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
cv2.destroyAllWindows()

# 保存点云（保留你的原注释）
# o3d.io.write_point_cloud("filtered_point_cloud.ply", pcd)
# print("点云已保存为 filtered_point_cloud.ply")