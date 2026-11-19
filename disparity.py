import cv2
import numpy as np

# 加载你的标定参数
data = np.load("stereo_calib_result.npz")
mtx_l, dist_l = data["mtx_l"], data["dist_l"]
mtx_r, dist_r = data["mtx_r"], data["dist_r"]
R, T, Q = data["R"], data["T"], data["Q"]

# 1. 打开你的双目相机（3840x1080）
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 2. 预计算校正映射
R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (1920,1080), R, T)
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (1920,1080), cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (1920,1080), cv2.CV_32FC1)

# 3. 创建视差匹配器
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*5,
    blockSize=9,
    P1=8*3*9**2,
    P2=32*3*9**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2
)

# 全局变量保存抓图后的图像
captured_left = None
captured_right = None
is_paused = False

while True:
    if not is_paused:
        ret, frame = cap.read()
        if not ret:
            break

        # 切开左右目
        left = frame[:, :1920]
        right = frame[:, 1920:]

        # 校正
        left_rect = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)

        # 实时显示校正后画面
        show_frame = np.hstack((left_rect, right_rect))
        show_frame = cv2.resize(show_frame, None, fx=0.5, fy=0.5)
        cv2.imshow("Camera (Space:Capture | Enter:Calc | Q:Quit)", show_frame)

    # 键盘控制
    key = cv2.waitKey(1) & 0xFF

    # 空格：抓图 + 暂停
    if key == ord(' '):
        print("✅ 已抓图，暂停画面")
        captured_left = left_rect.copy()
        captured_right = right_rect.copy()
        is_paused = True

    # 回车：计算视差
    elif key == 13:  # Enter 键
        if captured_left is not None and captured_right is not None:
            print("🔍 开始计算视差...")
            gray_l = cv2.cvtColor(captured_left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(captured_right, cv2.COLOR_BGR2GRAY)
            disparity = stereo.compute(gray_l, gray_r)

            # 显示视差图
            disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            disp_show = cv2.resize(disp_norm, None, fx=0.5, fy=0.5)
            cv2.imshow("Disparity Result", disp_show)
            print("✅ 视差计算完成！")
        else:
            print("❌ 请先按空格抓图！")

    # Q：退出
    elif key == ord('q'):
        print("👋 退出程序")
        break

cap.release()
cv2.destroyAllWindows()