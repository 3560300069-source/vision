r"""单对照片测距：在图上点两个点，直接给出两点间的真实距离（单位 mm）。

用法：
    python measure_pair.py                 # 默认第 1 对，弹窗点选
    python measure_pair.py 38              # 第 38 对
    python measure_pair.py 38 480 600 520 640   # 不给 GUI，直接量两个像素(u1 v1 u2 v2)
    python measure_pair.py 38 --nogui      # 不弹窗，只打印棋盘/点云统计

鼠标操作：左键点两个点 → 控制台和窗口上显示两点 3D 坐标与距离；c=清除，q=退出。
坐标单位：米（点云）/ 毫米（打印）。
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEFT_DIR = os.path.join(HERE, 'pairs', 'left')
RIGHT_DIR = os.path.join(HERE, 'pairs', 'right')
OUT = os.path.join(HERE, 'results')
CACHE = os.path.join(OUT, 'corners.npz')
PATTERN = (9, 6)
SQUARE = 0.05          # 方格边长(米)：板宽 500 mm / 10 格
OUT_W = 960

args = [a for a in sys.argv[1:] if not a.startswith('--')]
NOGUI = ('--nogui' in sys.argv) or (not args and False)
PAIR = int(args[0]) if args else 38
PIX = [int(v) for v in args[1:5]] if len(args) >= 5 else None


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def key(p):
    return int(''.join(c for c in os.path.basename(p) if c.isdigit()))


def main():
    z = np.load(CACHE, allow_pickle=True)
    cL, cR, used = z['cL'], z['cR'], z['used']
    sizeL = tuple(int(v) for v in z['sizeL'])
    sizeR = tuple(int(v) for v in z['sizeR'])

    Ls = sorted([f for f in os.listdir(LEFT_DIR) if f.lower().endswith('.jpg')],
                key=lambda f: int(''.join(c for c in f if c.isdigit())))
    Rs = sorted([f for f in os.listdir(RIGHT_DIR) if f.lower().endswith('.jpg')],
                key=lambda f: int(''.join(c for c in f if c.isdigit())))

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE
    ol = [objp] * len(used)
    cl = [c.reshape(-1, 1, 2).astype(np.float32) for c in cL]
    cr = [c.reshape(-1, 1, 2).astype(np.float32) for c in cR]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    _, K1, D1, rv1, tv1 = cv2.calibrateCamera(ol, cl, sizeL, None, None, criteria=crit)
    _, K2, D2, rv2, tv2 = cv2.calibrateCamera(ol, cr, sizeR, None, None, criteria=crit)

    numsL = [int(''.join(c for c in f if c.isdigit())) for f in Ls]
    numsR = [int(''.join(c for c in f if c.isdigit())) for f in Rs]
    if PAIR in used.tolist():
        pos = PAIR - 1
        k = used.tolist().index(PAIR)
    elif PAIR in numsL:
        pos = numsL.index(PAIR)
        k = used.tolist().index(pos + 1)
    else:
        raise SystemExit('没有第 %d 对。可用文件号: %s' % (PAIR, [numsL[u - 1] for u in used.tolist()]))
    Rl, _ = cv2.Rodrigues(rv1[k])
    Rr, _ = cv2.Rodrigues(rv2[k])
    R = Rr @ Rl.T
    T = (tv2[k] - R @ tv1[k]).ravel()

    out_w = min(OUT_W, sizeL[0])
    out_h = int(round(out_w * sizeL[1] / sizeL[0]))
    f = 0.98 * K2[0, 0] * out_w / sizeR[0]
    R1, R2 = cv2.stereoRectify(K1, D1, K2, D2, sizeL, R, T,
                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)[0:2]

    # 把棋盘格平移到校正图中央（u = f*x + cx，所以 cx 要按偏移量调整）
    cx, cy = out_w / 2.0, out_h / 2.0
    _P1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
    _u1 = cv2.undistortPoints(cl[k], K1, D1, R=R1, P=_P1).reshape(-1, 2)
    _cu, _cv = float(np.median(_u1[:, 0])), float(np.median(_u1[:, 1]))
    _dx, _dy = cx - _cu, cy - _cv
    if abs(_dx) > 5 or abs(_dy) > 5:
        cx = float(min(max(cx + _dx, 60.0), out_w - 60.0))
        cy = float(min(max(cy + _dy, 60.0), out_h - 60.0))
        print('已把棋盘平移到画面中央: cx=%.0f cy=%.0f（偏移 %+.0f, %+.0f）' % (cx, cy, _dx, _dy))
    P1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
    P2 = np.array([[f, 0, cx, f * T[0]], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
    Q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1.0 / T[0], 0]], np.float64)

    iml = imread_u(os.path.join(LEFT_DIR, Ls[pos]))
    imr = imread_u(os.path.join(RIGHT_DIR, Rs[pos]))
    if (imr.shape[1], imr.shape[0]) != sizeR:
        imr = cv2.resize(imr, sizeR, interpolation=cv2.INTER_AREA)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (out_w, out_h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (out_w, out_h), cv2.CV_32FC1)
    rl = cv2.remap(iml, m1[0], m1[1], cv2.INTER_LINEAR)
    rr = cv2.remap(imr, m2[0], m2[1], cv2.INTER_LINEAR)
    gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)

    nd_req = float(f) * float(np.linalg.norm(T)) / 0.5
    if (not np.isfinite(nd_req)) or nd_req <= 0:
        print('视差范围估算异常，按 400 处理')
        nd_req = 400.0
    nd = 16 * (int(nd_req) // 16 + 1)
    if nd > 512:
        nd = 512
    disp = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=9,
        P1=8 * 3 * 81, P2=32 * 3 * 81, disp12MaxDiff=2, uniquenessRatio=5,
        speckleWindowSize=300, speckleRange=2, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY).compute(gl, gr).astype(np.float32) / 16.0
    disp[(disp <= 0) | (disp >= nd - 1)] = 0

    pts_rect = cv2.reprojectImageTo3D(disp, Q)
    with np.errstate(invalid='ignore', divide='ignore'):
        pts = (pts_rect.reshape(-1, 3) @ R1).reshape(pts_rect.shape)     # 回转左相机坐标系
    b = float(np.linalg.norm(T))
    print('=' * 64)
    print('第 %d 对（left/%s <-> right/%s）' % (PAIR, Ls[pos], Rs[pos]))
    print('基线 %.1f mm，焦距 %.1f px，方格 %.0f mm' % (b * 1000, f, SQUARE * 1000))

    # 棋盘距离（用棋盘角点的视差直接算，比稠密匹配稳）
    u1 = cv2.undistortPoints(cl[k], K1, D1, R=R1, P=P1).reshape(-1, 2)
    u2 = cv2.undistortPoints(cr[k], K2, D2, R=R2, P=P2).reshape(-1, 2)
    d_board = float(np.mean(u1[:, 0] - u2[:, 0]))
    Z_board = abs(f * T[0]) / max(d_board, 1e-6)
    print('棋盘格到左相机距离: %.0f mm  (视差 %.1f px)' % (Z_board * 1000, d_board))
    bx = u1[:, 0].max() - u1[:, 0].min()
    by = u1[:, 1].max() - u1[:, 1].min()
    print('棋盘在图中跨度: %.0f x %.0f px  →  推算板面尺寸 %.0f x %.0f mm'
          % (bx, by, bx * Z_board / f * 1000, by * Z_board / f * 1000))
    print('（板宽应为 10 格 = %.0f mm；若实测板宽差异大，说明方格边长设错了）' % (SQUARE * 10 * 1000))

    if '--corners' in sys.argv:
        ncol = PATTERN[0]
        cand = [(0, '左上'), (ncol - 1, '右上'),
                (len(u1) - ncol, '左下'), (len(u1) - 1, '右下')]
        print('棋盘四角在校正图里的像素坐标（可直接用于命令行量测）:')
        for idx, name in cand:
            print('  %s: u=%.0f  v=%.0f' % (name, u1[idx, 0], u1[idx, 1]))
        diag = ((PATTERN[0] - 1) ** 2 + (PATTERN[1] - 1) ** 2) ** 0.5 * SQUARE * 1000
        print('  板对角线应有长度: %.1f mm（%d 格 x %.0f mm 与 %d 格 x %.0f mm 的斜边）'
              % (diag, PATTERN[0] - 1, SQUARE * 1000, PATTERN[1] - 1, SQUARE * 1000))
        sys.exit(0)

    valid = disp > 0
    Zc = pts[:, :, 2]
    okz = valid & np.isfinite(Zc) & (Zc > 0.05)
    if okz.any():
        print('有效深度点 %.1f%%，深度中位 %.0f mm' % (valid.mean() * 100, np.median(Zc[okz]) * 1000))

    def point3d(u, v, win=6):
        """取 (u,v) 附近有效视差的中位数，返回相机坐标系 3D 点（米）。"""
        u0, u1_ = max(0, u - win), min(out_w, u + win + 1)
        v0, v1_ = max(0, v - win), min(out_h, v + win + 1)
        patch = disp[v0:v1_, u0:u1_]
        pv = patch[patch > 0]
        if pv.size == 0:
            return None, None
        d = float(np.median(pv))
        Z = f * abs(T[0]) / d
        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f
        p_rect = np.array([X, Y, Z])
        return R1.T @ p_rect, d

    def report(u1_, v1_, u2_, v2_):
        p1, d1 = point3d(u1_, v1_)
        p2, d2 = point3d(u2_, v2_)
        if p1 is None or p2 is None:
            print('  所选点没有有效视差，换个位置（避开纯色/反光面）')
            return
        dist = np.linalg.norm(p1 - p2) * 1000
        print('  点1 (u=%d,v=%d): d=%.1fpx  相机距离 %.0f mm  坐标(%.0f, %.0f, %.0f) mm'
              % (u1_, v1_, d1, np.linalg.norm(p1) * 1000, p1[0] * 1000, p1[1] * 1000, p1[2] * 1000))
        print('  点2 (u=%d,v=%d): d=%.1fpx  相机距离 %.0f mm  坐标(%.0f, %.0f, %.0f) mm'
              % (u2_, v2_, d2, np.linalg.norm(p2) * 1000, p2[0] * 1000, p2[1] * 1000, p2[2] * 1000))
        print('  ==> 两点距离: %.1f mm   （水平 %.1f，垂直 %.1f，前后 %.1f）'
              % (dist, (p2[0] - p1[0]) * 1000, (p2[1] - p1[1]) * 1000, (p2[2] - p1[2]) * 1000))

    if PIX:
        report(*PIX)
        return

    if NOGUI:
        print('未指定像素坐标，跳过交互（可用: python measure_pair.py %d u1 v1 u2 v2）' % PAIR)
        return

    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            if len(clicks) > 2:
                clicks.pop(0)
            if len(clicks) == 2:
                report(clicks[0][0], clicks[0][1], clicks[1][0], clicks[1][1])

    win = 'measure pair %d  (click 2 points, c=clear, q=quit)' % PAIR
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)
    except Exception as e:
        print('无法开窗口(%s)，请用像素坐标模式' % e)
        return
    while True:
        show = rl.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(show, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(show, 'P%d' % (i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(clicks) == 2:
            cv2.line(show, clicks[0], clicks[1], (0, 255, 255), 2)
        cv2.imshow(win, show)
        kk = cv2.waitKey(20) & 0xFF
        if kk == ord('q') or kk == 27:
            break
        if kk == ord('c'):
            clicks.clear()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
