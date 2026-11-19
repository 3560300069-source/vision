import cv2
import numpy as np

# ===================== Configuration Parameters =====================
IMG_PATH = r'F:\Automatic-Number-Plate-Detection-main\Automatic-Number-Plate-Detection-main\vision3d\test_images\stereo_01.png'
CALIB_FILE = "stereo_calib_result.npz"
W, H = 1920, 1080

# Industrial measurement core parameters
NEIGHBORHOOD_SIZE = 15  # Sampling window size around marked point (odd, 11-21 recommended)
MIN_VALID_PIXELS = 50   # Minimum valid pixels, measurement invalid below this
OUTLIER_THRESHOLD = 3.0 # 3σ outlier rejection threshold
MIN_DISPARITY = 16
MAX_DISPARITY = 400
# ====================================================================

# Global variables
click_point = None
depth_result = ""
confidence = 0.0

# Load calibration parameters
data = np.load(CALIB_FILE)
mtx_l = data["mtx_l"]
dist_l = data["dist_l"]
mtx_r = data["mtx_r"]
dist_r = data["dist_r"]
R = data["R"]
T = data["T"]
Q = data["Q"]

# Epipolar rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, (W, H), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (W, H), cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (W, H), cv2.CV_32FC1)

# Read and rectify image
full = cv2.imread(IMG_PATH)
if full is None:
    print("❌ Error: Failed to read image!")
    exit()

left_raw = full[:, :W]
right_raw = full[:, W:]
left_rect = cv2.remap(left_raw, map1_l, map2_l, cv2.INTER_LINEAR)
right_rect = cv2.remap(right_raw, map1_r, map2_r, cv2.INTER_LINEAR)

# Precompute global disparity map (Industrial SGBM + lightweight filtering)
print("Precomputing disparity map...")
left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

# Initialize SGBM matcher (optimized parameters)
sgbm = cv2.StereoSGBM_create(
    minDisparity=MIN_DISPARITY,
    numDisparities=384,
    blockSize=9,
    P1=8 * 3 * 9 * 9,
    P2=32 * 3 * 9 * 9,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=150,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute and filter disparity map
disparity = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
# Lightweight bilateral filter (preserve edges + smooth noise)
disparity_filtered = cv2.bilateralFilter(disparity, d=5, sigmaColor=15, sigmaSpace=15)
print("✅ Disparity map precomputed successfully")


# ===================== Mouse Callback Function =====================
def mouse_callback(event, x, y, flags, param):
    global click_point, depth_result, confidence
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert display coordinates to original image coordinates
        x_original = x * 2
        y_original = y * 2
        click_point = (x_original, y_original)
        print(f"\n📍 Marked point: ({x_original}, {y_original})")
        # Automatically calculate depth
        depth_result, confidence = calculate_industrial_depth(x_original, y_original)


# ===================== Industrial Depth Calculation Core Function =====================
def calculate_industrial_depth(x, y):
    """
    Industrial-grade depth calculation method:
    1. Multi-pixel neighborhood sampling
    2. Invalid disparity filtering
    3. 3σ statistical outlier rejection
    4. Median robust estimation
    5. Confidence evaluation
    """
    # Calculate neighborhood ROI boundaries
    half_win = NEIGHBORHOOD_SIZE // 2
    x1 = max(0, x - half_win)
    x2 = min(W, x + half_win + 1)
    y1 = max(0, y - half_win)
    y2 = min(H, y + half_win + 1)

    # Extract disparities in ROI
    roi_disp = disparity_filtered[y1:y2, x1:x2]

    # Step 1: Filter invalid disparities
    valid_mask = (roi_disp > MIN_DISPARITY) & (roi_disp < MAX_DISPARITY)
    valid_disps = roi_disp[valid_mask]

    if len(valid_disps) < MIN_VALID_PIXELS:
        return "Invalid: Too few valid pixels", 0.0

    # Step 2: 3σ principle outlier rejection
    mean_disp = np.mean(valid_disps)
    std_disp = np.std(valid_disps)
    outlier_mask = np.abs(valid_disps - mean_disp) < OUTLIER_THRESHOLD * std_disp
    filtered_disps = valid_disps[outlier_mask]

    if len(filtered_disps) < MIN_VALID_PIXELS // 2:
        return "Invalid: Too much noise", 0.0

    # Step 3: Median robust estimation (3x more robust than mean)
    final_disp = np.median(filtered_disps)

    # Step 4: Calculate depth
    dummy = np.zeros((H, W), np.float32)
    dummy[y, x] = final_disp
    depth = cv2.reprojectImageTo3D(dummy, Q)[y, x, 2]

    if depth <= 0 or depth > 10.0:
        return "Invalid: Depth out of range", 0.0

    # Step 5: Calculate confidence (valid pixel ratio)
    confidence = len(filtered_disps) / (roi_disp.size) * 100

    # Calculate theoretical error (based on your camera parameters)
    focal_length = mtx_l[0, 0]
    baseline = abs(T[0][0])
    theoretical_error = (depth ** 2) / (focal_length * baseline) * 0.1  # Δd=0.1 pixel

    return (f"Depth: {depth * 1000:.1f} mm\n"
            f"Theo Error: ±{theoretical_error * 1000:.1f} mm\n"
            f"Confidence: {confidence:.1f}%"), confidence


# ===================== Drawing Function =====================
def draw_interface(img):
    # Draw marked point and neighborhood window
    if click_point is not None:
        x, y = click_point
        half_win = NEIGHBORHOOD_SIZE // 2
        # Draw center point
        cv2.circle(img, (x // 2, y // 2), 5, (0, 0, 255), -1)
        # Draw neighborhood window
        cv2.rectangle(img,
                      ((x - half_win) // 2, (y - half_win) // 2),
                      ((x + half_win + 1) // 2, (y + half_win + 1) // 2),
                      (0, 255, 0), 2)

    # Draw measurement results
    if depth_result:
        y_offset = 60
        for line in depth_result.split('\n'):
            cv2.putText(img, line, (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            y_offset += 40

    # Draw operation instructions
    cv2.putText(img, "Left click to mark | Q to quit", (30, H // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ===================== Main Loop =====================
cv2.namedWindow("Industrial Depth Measurement")
cv2.setMouseCallback("Industrial Depth Measurement", mouse_callback)

print("\n✅ Measurement tool started")
print("Instructions:")
print("  - Left click anywhere to mark measurement point")
print("  - Depth, error and confidence will be displayed automatically")
print("  - Press C to clear mark | Press Q to exit")

while True:
    display_img = cv2.resize(left_rect.copy(), (W // 2, H // 2))
    draw_interface(display_img)

    cv2.imshow("Industrial Depth Measurement", display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear mark
        click_point = None
        depth_result = ""
        confidence = 0.0

cv2.destroyAllWindows()