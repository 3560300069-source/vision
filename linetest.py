import cv2
import numpy as np

# ===================== 配置参数 =====================
IMG_PATH = r'F:\Automatic-Number-Plate-Detection-main\Automatic-Number-Plate-Detection-main\vision3d\test_images\stereo_01.png'
CALIB_FILE = "stereo_calib_result.npz"
W, H = 1920, 1080
# ====================================================

# 加载标定
data = np.load(CALIB_FILE)
mtx_l = data["mtx_l"]
dist_l = data["dist_l"]
mtx_r = data["mtx_r"]
dist_r = data["dist_r"]
R = data["R"]
T = data["T"]
Q = data["Q"]

# 极线校正
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, (W, H), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (W, H), cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (W, H), cv2.CV_32FC1)

# 读取图像
full = cv2.imread(IMG_PATH)
left_rect = cv2.remap(full[:, :W], map1_l, map2_l, cv2.INTER_LINEAR)
right_rect = cv2.remap(full[:, W:], map1_r, map2_r, cv2.INTER_LINEAR)

# ========== 四根线 ==========
vline1 = 400   # 左竖线 绿 A D
vline2 = 600   # 右竖线 绿 4 6
hline1 = 250   # 上横线 红 W S
hline2 = 450   # 下横线 红 2 8

calc_result = ""
step = 2

# ========== 绘制 ==========
def draw_lines(img):
    cv2.line(img, (vline1, 0), (vline1, H), (0,255,0), 2)
    cv2.line(img, (vline2, 0), (vline2, H), (0,255,0), 2)
    cv2.line(img, (0, hline1), (W, hline1), (0,0,255), 2)
    cv2.line(img, (0, hline2), (W, hline2), (0,0,255), 2)

# ========== 【区域内精细匹配 + 去干扰】 ==========
def calc_roi_size():
    x1 = min(vline1, vline2)
    x2 = max(vline1, vline2)
    y1 = min(hline1, hline2)
    y2 = max(hline1, hline2)

    disps = []
    win = 15
    for y in range(y1 + win, y2 - win, 5):
        rowL = left_rect[y, :, 0]
        rowR = right_rect[y, :, 0]
        for x in range(x1 + win, x2 - win, 5):
            patch = rowL[x-win:x+win]
            corr = cv2.matchTemplate(rowR, patch, cv2.TM_CCOEFF_NORMED)
            disp = x - np.argmax(corr)
            disps.append(disp)

    if len(disps) < 10:
        return "ROI太小"

    disp = np.median(disps)
    dummy = np.zeros((H,W), np.float32)
    cy = (y1+y2)//2
    dummy[cy, W//2] = disp
    z = cv2.reprojectImageTo3D(dummy, Q)[cy, W//2, 2]

    if z <= 0:
        return "深度错误"

    pix_w = abs(vline2 - vline1)
    pix_h = abs(hline2 - hline1)
    pp = z / mtx_l[0,0]
    w_mm = pix_w * pp * 1000
    h_mm = pix_h * pp * 1000

    return f"W:{w_mm:.1f}mm  H:{h_mm:.1f}mm"

# ========== 主循环 ==========
while True:
    frame = left_rect.copy()
    draw_lines(frame)

    if calc_result:
        cv2.putText(frame, calc_result, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,0),3)

    cv2.imshow("3D Measure", cv2.resize(frame, (960,540)))
    key = cv2.waitKey(1) & 0xFF

    # === 竖线 ===
    if key == ord('a'): vline1 -= step
    elif key == ord('d'): vline1 += step
    elif key == ord('4'): vline2 -= step
    elif key == ord('6'): vline2 += step

    # === 横线 ===
    elif key == ord('w'): hline1 -= step
    elif key == ord('s'): hline1 += step
    elif key == ord('2'): hline2 += step
    elif key == ord('8'): hline2 -= step

    # === 计算 / 退出 ===
    elif key == 13: calc_result = calc_roi_size()
    elif key == ord('q'): break

cv2.destroyAllWindows()