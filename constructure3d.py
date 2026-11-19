import cv2
import numpy as np
import open3d as o3d

# ===================== Configuration Parameters =====================
IMG_PATH = r'F:\Automatic-Number-Plate-Detection-main\Automatic-Number-Plate-Detection-main\vision3d\test_images\stereo_01.png'
CALIB_FILE = "stereo_calib_result.npz"
W, H = 1920, 1080

# Disparity parameters (optimized)
MIN_DISPARITY = 20
NUM_DISPARITIES = 416
BLOCK_SIZE = 11
SGBM_P1 = int(8 * 3 * BLOCK_SIZE ** 2)
SGBM_P2 = int(32 * 3 * BLOCK_SIZE ** 2)
DISP12_MAX_DIFF = 3
UNIQUENESS_RATIO = 12
SPECKLE_WINDOW_SIZE = 200
SPECKLE_RANGE = 40

# Filter switches (enable based on effect)
ENABLE_MEDIAN_FILTER = True       # Median filter: remove salt-and-pepper noise
ENABLE_GAUSSIAN_FILTER = False    # Gaussian filter: overall smoothing
ENABLE_BILATERAL_FILTER = True    # Bilateral filter: edge-preserving smoothing
ENABLE_MORPHOLOGY_FILTER = True   # Morphological filter: fill small holes
ENABLE_HOLE_FILLING = True        # Hole filling: repair large black areas

# Filter parameters
MEDIAN_KERNEL = 3                 # Median filter kernel size (odd)
GAUSSIAN_KERNEL = 3               # Gaussian filter kernel size (odd)
BILATERAL_D = 5                   # Bilateral filter diameter
BILATERAL_SIGMA_COLOR = 50        # Color space sigma
BILATERAL_SIGMA_SPACE = 50        # Coordinate space sigma
MORPHOLOGY_KERNEL = np.ones((3,3), np.uint8)  # Morphological kernel
# =====================================================================

# 1. Load calibration parameters
print("="*60)
print("🔍 Calibration Parameters:")
data = np.load(CALIB_FILE)
mtx_l = data["mtx_l"]
dist_l = data["dist_l"]
mtx_r = data["mtx_r"]
dist_r = data["dist_r"]
R = data["R"]
T = data["T"]

focal_length = mtx_l[0,0]
baseline = abs(T[0][0])
print(f"Focal length fx: {focal_length:.2f} pixels")
print(f"Baseline B: {baseline*1000:.1f} mm")
print(f"Theoretical detection range: {focal_length*baseline/(MIN_DISPARITY+NUM_DISPARITIES):.2f}m ~ {focal_length*baseline/MIN_DISPARITY:.2f}m")
print("="*60)

# 2. Epipolar rectification
print("Performing epipolar rectification...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, (W, H), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (W, H), cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (W, H), cv2.CV_32FC1)

# 3. Read and rectify images
print("Reading images...")
full_img = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)
if full_img is None:
    print("❌ Failed to read image!")
    exit()

left_raw = full_img[:, :W]
right_raw = full_img[:, W:]

left_rect = cv2.remap(left_raw, map1_l, map2_l, cv2.INTER_LINEAR)
right_rect = cv2.remap(right_raw, map1_r, map2_r, cv2.INTER_LINEAR)

left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

# 4. Compute raw disparity map
print("Computing disparity map...")
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

# 5. Disparity filtering pipeline (core optimization)
print("Applying disparity filtering...")
filtered_disp = disparity.copy()

# Step 1: Median filter (remove salt-and-pepper noise)
if ENABLE_MEDIAN_FILTER:
    filtered_disp = cv2.medianBlur(filtered_disp, MEDIAN_KERNEL)

# Step 2: Morphological operations (remove small bright spots, fill small dark spots)
if ENABLE_MORPHOLOGY_FILTER:
    filtered_disp = cv2.morphologyEx(filtered_disp, cv2.MORPH_CLOSE, MORPHOLOGY_KERNEL)
    filtered_disp = cv2.morphologyEx(filtered_disp, cv2.MORPH_OPEN, MORPHOLOGY_KERNEL)

# Step 3: Bilateral filter (smooth surfaces while preserving edges)
if ENABLE_BILATERAL_FILTER:
    filtered_disp = cv2.bilateralFilter(filtered_disp, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

# Step 4: Gaussian filter (overall smoothing)
if ENABLE_GAUSSIAN_FILTER:
    filtered_disp = cv2.GaussianBlur(filtered_disp, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)

# Step 5: Hole filling (repair invalid disparity regions)
if ENABLE_HOLE_FILLING:
    mask = (filtered_disp <= MIN_DISPARITY) | (filtered_disp >= MIN_DISPARITY + NUM_DISPARITIES)
    # Fill holes using neighborhood interpolation
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    filtered_disp[mask] = cv2.inpaint(filtered_disp, dilated_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)[mask]

# Disparity statistics
valid_disp = filtered_disp[(filtered_disp > MIN_DISPARITY) & (filtered_disp < MIN_DISPARITY + NUM_DISPARITIES)]
print(f"\nFiltered Disparity Statistics:")
print(f"  Min disparity: {np.min(valid_disp):.2f}")
print(f"  Max disparity: {np.max(valid_disp):.2f}")
print(f"  Mean disparity: {np.mean(valid_disp):.2f}")
print(f"  Disparity std: {np.std(valid_disp):.2f}")
print("="*60)

# Display comparison
disp_raw_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
disp_filtered_normalized = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
disp_raw_color = cv2.applyColorMap(disp_raw_normalized, cv2.COLORMAP_JET)
disp_filtered_color = cv2.applyColorMap(disp_filtered_normalized, cv2.COLORMAP_JET)

comparison = np.hstack((disp_raw_color, disp_filtered_color))
cv2.putText(comparison, "Raw Disparity", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
cv2.putText(comparison, "Filtered Disparity", (W+50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
cv2.imshow("Disparity Comparison", cv2.resize(comparison, (1920, 540)))
cv2.waitKey(3000)

# 6. Generate 3D point cloud
print("Generating 3D point cloud...")
points_3d = cv2.reprojectImageTo3D(filtered_disp, Q)

mask = (filtered_disp > MIN_DISPARITY) & (filtered_disp < MIN_DISPARITY + NUM_DISPARITIES)
valid_points = points_3d[mask]
valid_colors = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)[mask] / 255.0

# Only show points between 0.9m and 1.6m (40cm around 1m)
min_distance = 0.9  # 1m - 40cm
max_distance = 1.6  # 1m + 40cm
distance_mask = (np.linalg.norm(valid_points, axis=1) < max_distance) & \
                (np.linalg.norm(valid_points, axis=1) > min_distance)
valid_points = valid_points[distance_mask]
valid_colors = valid_colors[distance_mask]

# Point cloud downsampling (greatly improves performance, fixes GLFW errors)
print("Downsampling point cloud...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)
pcd.colors = o3d.utility.Vector3dVector(valid_colors)
pcd = pcd.voxel_down_sample(voxel_size=0.003)  # 3mm voxel, balance quality and speed

print(f"Final valid points: {len(pcd.points):,}")
print(f"Actual depth range: {np.min(np.asarray(pcd.points)[:,2]):.2f} ~ {np.max(np.asarray(pcd.points)[:,2]):.2f} m")
print("="*60)

# 7. Visualization (fixed GLFW context error)
print("Opening visualization window...")
print("Controls: Left-click to rotate, Right-click to pan, Scroll to zoom, Q to exit")

# Critical setting to fix GLFW context errors
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Filtered 3D Point Cloud", width=1200, height=800, visible=True)

# Set rendering options
opt = vis.get_render_option()
opt.background_color = np.array([0, 0, 0])
opt.point_size = 2.0
opt.show_coordinate_frame = True
opt.light_on = True

vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
cv2.destroyAllWindows()

# Save point cloud
# o3d.io.write_point_cloud("filtered_point_cloud.ply", pcd)
# print("Point cloud saved as filtered_point_cloud.ply")