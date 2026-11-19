r"""对某一对左右照片：校正 -> 视差(SGBM) -> 点云(PLY) -> 三视图预览。

用法：
    python run_pair.py            # 默认重建 41.jpg，并打开两点测量
    python run_pair.py 11         # 第 11 对（文件名 11.jpg）
    python run_pair.py 11 1440    # 指定校正后宽度

用 --object-folder 测量 object/left.jpg 和 object/right.jpg，也可用 --object LEFT RIGHT。
对象模式复用通过极线检查的固定双目标定。可加 --no-gui 仅导出文件。
输出：results\rect_left_XX.png / rect_right_XX.png / disp_XX.png /
      pointcloud_pairXX.ply / cloud_view1_XX.png / cloud_view2_XX.png
点云单位是米，可直接用 MeshLab / CloudCompare / open3d 打开。
"""
import argparse, cv2, numpy as np, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LEFT_DIR = os.path.join(HERE, 'pairs', 'left')
RIGHT_DIR = os.path.join(HERE, 'pairs', 'right')
OUT = os.path.join(HERE, 'results')
CACHE = os.path.join(OUT, 'corners.npz')
PATTERN = (9, 6)
SQUARE = 0.05

DEFAULT_PAIR = 41


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def key(p):
    return int(''.join(c for c in os.path.basename(p) if c.isdigit()))


def demo_rectification(PAIR, width):
    Ls = sorted(glob.glob(os.path.join(LEFT_DIR, '*.jpg')), key=key)
    Rs = sorted(glob.glob(os.path.join(RIGHT_DIR, '*.jpg')), key=key)
    numsL = [key(p) for p in Ls]
    if numsL != [key(p) for p in Rs] or len(numsL) != len(set(numsL)):
        raise SystemExit('左右照片须有一一对应的文件名编号')
    if PAIR not in numsL:
        raise SystemExit(f'找不到 {PAIR}.jpg，可用文件名编号：{numsL}')
    pos = numsL.index(PAIR)
    signature = '#'.join('|'.join(f'{os.path.basename(p)}:{os.path.getsize(p)}' for p in files)
                         for files in (Ls, Rs))
    refresh = not os.path.isfile(CACHE)
    if not refresh:
        with np.load(CACHE, allow_pickle=True) as cached:
            refresh = 'sig' not in cached.files or str(cached['sig']) != signature
    if refresh:
        print('正在标定左右相机……', flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, 'stereo_calib_pairs.py'),
                        LEFT_DIR, RIGHT_DIR], check=True)
    with np.load(CACHE, allow_pickle=True) as cached:
        cL, cR, used = cached['cL'], cached['cR'], cached['used']
        sizeL = tuple(int(v) for v in cached['sizeL'])
        sizeR = tuple(int(v) for v in cached['sizeR'])
    if pos + 1 not in used:
        raise SystemExit(f'{PAIR}.jpg 没有检测到棋盘格')
    print(f'重建图对 {PAIR}.jpg', flush=True)

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE
    ol = [objp] * len(used)
    cl = [c.reshape(-1, 1, 2).astype(np.float32) for c in cL]
    cr = [c.reshape(-1, 1, 2).astype(np.float32) for c in cR]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    _, K1, D1, rv1, tv1 = cv2.calibrateCamera(ol, cl, sizeL, None, None, criteria=crit)
    _, K2, D2, rv2, tv2 = cv2.calibrateCamera(ol, cr, sizeR, None, None, criteria=crit)

    k = used.tolist().index(pos + 1)
    Rl, _ = cv2.Rodrigues(rv1[k]); Rr, _ = cv2.Rodrigues(rv2[k])
    R = Rr @ Rl.T
    T = (tv2[k] - R @ tv1[k]).ravel()      # 该图对自身的相对位姿
    B = float(np.linalg.norm(T))
    print(f'第 {PAIR} 对: 基线 {B*1000:.1f} mm, T(mm)={np.round(T*1000,1)}')

    R1, R2 = cv2.stereoRectify(K1, D1, K2, D2, sizeL, R, T,
                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)[0:2]
    out_w = min(width, sizeL[0])
    if out_w < 128:
        raise SystemExit("校正图宽度至少为 128 像素")
    out_h = int(round(out_w * sizeL[1] / sizeL[0]))
    f = 0.98 * K2[0, 0] * out_w / sizeR[0]          # 以低分辨率那台的原生焦距为准
    cx, cy = out_w / 2.0, out_h / 2.0
    P1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
    P2 = np.array([[f, 0, cx, f * T[0]], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
    Q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1.0 / T[0], 0]], np.float64)
    print(f'校正输出 {out_w}x{out_h}, f={f:.1f}')

    u1 = cv2.undistortPoints(cl[k], K1, D1, R=R1, P=P1).reshape(-1, 2)
    u2 = cv2.undistortPoints(cr[k], K2, D2, R=R2, P=P2).reshape(-1, 2)
    dy = np.abs(u1[:, 1] - u2[:, 1])
    db = float(np.mean(u1[:, 0] - u2[:, 0]))
    print(f'极线自检 |dy|={dy.mean():.2f}px (最大 {dy.max():.2f}); 棋盘视差 {db:.0f}px -> '
          f'深度 {abs(f*T[0])/max(db,1)*1000:.0f} mm')

    return (Ls[pos], Rs[pos]), (K1, D1, K2, D2, R1, R2, P1, P2, Q), sizeL, sizeR, (out_w, out_h)


def fixed_rectification(width):
    calibration_path = os.path.join(OUT, 'stereo_calib_result_pairs.npz')
    report_path = os.path.join(OUT, 'calib_report.json')
    if not os.path.isfile(calibration_path) or not os.path.isfile(report_path):
        subprocess.run([sys.executable, os.path.join(HERE, 'stereo_calib_pairs.py')], check=True)

    with np.load(calibration_path) as model:
        sizeL = tuple(int(v) for v in model['img_size'])
        sizeR = tuple(int(v) for v in model['right_img_size'])
        K1, D1, K2, D2, R, T = (model[key] for key in
                                 ('mtx_l', 'dist_l', 'mtx_r', 'dist_r', 'R', 'T'))
    baseline_mm = 1000 * float(np.linalg.norm(T))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, sizeL, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    width = min(width, sizeL[0])
    height = round(width * sizeL[1] / sizeL[0])
    sx, sy = width / sizeL[0], height / sizeL[1]
    with open(report_path, encoding='utf-8') as file:
        report = json.load(file)
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    error_axis = 0 if vertical else 1
    residual = float(np.percentile(report['epipolar_err_px'], 90) * (sx, sy)[error_axis])
    print(f'Fixed stereo: RMS {report["rms_stereo"]:.2f} px, '
          f'fx left/right {K1[0, 0]:.1f}/{K2[0, 0]:.1f} px, baseline {baseline_mm:.1f} mm, '
          f'rectified epipolar error P90 {residual:.2f} px.' ,
          flush=True)
    if not np.isfinite(residual) or residual > 1.0:
        print('Warning: calibration report exceeds the epipolar-error threshold. '
              'Continuing in trial mode; measurements may be inaccurate.', flush=True)

    scale = np.diag([sx, sy, 1.0])
    disparity_scale = sy if vertical else sx
    # Convert output coordinates and disparity to the calibration pixel grid.
    Q = Q @ np.diag([1 / sx, 1 / sy, 1 / disparity_scale, 1.0])
    return (K1, D1, K2, D2, R1, R2, scale @ P1, scale @ P2, Q), sizeL, sizeR, (width, height)


def repair_disparity(disparity):
    """Fill only small enclosed holes with a consistent surrounding disparity."""
    missing = disparity == 0
    height, width = missing.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
    left, top, box_w, box_h, area = stats.T
    enclosed = ((area <= 16) & (left > 0) & (top > 0)
                & (left + box_w < width) & (top + box_h < height))
    enclosed[0] = False
    holes = enclosed[labels]
    if not holes.any():
        return disparity, holes

    kernel = np.ones((5, 5), np.uint8)
    high = cv2.dilate(disparity, kernel)
    low = cv2.erode(np.where(missing, np.finfo(np.float32).max, disparity), kernel)
    count = cv2.boxFilter((~missing).astype(np.float32), -1, (5, 5), normalize=False)
    total = cv2.boxFilter(disparity, -1, (5, 5), normalize=False)
    # A disparity jump across the hole can indicate an occlusion or surface boundary.
    eligible = (count >= 8) & (high - low <= 2.0)
    rejected = np.unique(labels[holes & ~eligible])
    enclosed[rejected] = False
    repaired = enclosed[labels]
    output = disparity.copy()
    output[repaired] = total[repaired] / count[repaired]
    return output, repaired


def main():
    parser = argparse.ArgumentParser(description='单组双目点云与两点测量')
    parser.add_argument('pair', type=int, nargs='?', help='指定棋盘图对的照片文件名编号')
    parser.add_argument('width', type=int, nargs='?', default=960, help='校正图宽度')
    parser.add_argument('--no-gui', action='store_true', help='只导出点云和图片')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--object', nargs=2, metavar=('LEFT', 'RIGHT'))
    inputs.add_argument('--object-folder', action='store_true',
                        help='measure object/left.jpg and object/right.jpg')
    args = parser.parse_args()
    if args.width < 128:
        parser.error('Measurement width must be at least 128 pixels')
    PAIR = args.pair if args.pair is not None else DEFAULT_PAIR

    default_object = tuple(os.path.join(HERE, 'object', name) for name in ('left.jpg', 'right.jpg'))
    object_paths = default_object if args.object_folder else args.object

    if object_paths:
        if not all(os.path.isfile(path) for path in object_paths):
            raise SystemExit('Object measurement requires both left and right images.')
        paths = object_paths
        rect, sizeL, sizeR, output_size = fixed_rectification(args.width)
        PAIR = 'object'
        suffix = PAIR
    else:
        paths, rect, sizeL, sizeR, output_size = demo_rectification(PAIR, args.width)
        suffix = f'{PAIR:02d}'

    K1, D1, K2, D2, R1, R2, P1, P2, Q = rect
    out_w, out_h = output_size
    iml, imr = (imread_u(path) for path in paths)
    if iml is None or imr is None:
        raise SystemExit('Could not decode the left or right image.')
    if object_paths and (iml.shape[1::-1] != sizeL or imr.shape[1::-1] != sizeR):
        raise SystemExit(f'Use calibration image sizes: left {sizeL}, right {sizeR}.')
    if imr.shape[1::-1] != sizeR:
        imr = cv2.resize(imr, sizeR, interpolation=cv2.INTER_AREA)
    m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (out_w, out_h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (out_w, out_h), cv2.CV_32FC1)
    rl = cv2.remap(iml, m1[0], m1[1], cv2.INTER_LINEAR)
    rr = cv2.remap(imr, m2[0], m2[1], cv2.INTER_LINEAR)
    gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)

    # 视差：两台相机清晰度不同，blockSize 取大一些更稳
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    projection_offset = P2[1 if vertical else 0, 3]
    if projection_offset == 0:
        raise SystemExit('Calibration has zero stereo baseline.')
    if vertical:
        gl, gr = gl.T, gr.T
    reverse_disparity = projection_offset > 0
    if reverse_disparity:
        gl, gr = gl[:, ::-1], gr[:, ::-1]
    nd = min(16 * 44, ((gl.shape[1] - 16) // 16) * 16)
    bs = 9
    disp_l = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=bs,
        P1=8 * 3 * bs ** 2, P2=32 * 3 * bs ** 2, disp12MaxDiff=2,
        uniquenessRatio=5, speckleWindowSize=300, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY).compute(gl, gr)
    raw_disparity = disp_l.astype(np.float32) / 16.0
    matched = (raw_disparity > 0) & (raw_disparity < nd - 1)
    print(f'有效视差 {matched.mean()*100:.1f}%')

    median = cv2.medianBlur(np.where(matched, raw_disparity, 0), 5)
    mask = (matched & (median > 0)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 500
    keep[0] = False
    disp = np.where(keep[labels], median, 0)

    def image_coordinates(array):
        if reverse_disparity:
            array = array[:, ::-1]
        return array.T if vertical else array

    disp, repaired = repair_disparity(disp)
    disparity_sign = -1 if reverse_disparity else 1
    disp = image_coordinates(disp) * disparity_sign
    repaired = image_coordinates(repaired)
    print(f'滤波并修复后有效视差 {(disp != 0).mean()*100:.1f}%，填补 {repaired.sum()} 像素')
    pts_rect = cv2.reprojectImageTo3D(disp, Q)
    with np.errstate(invalid='ignore', divide='ignore'):
        pts = (pts_rect.reshape(-1, 3) @ R1).reshape(pts_rect.shape)   # 回转原始左相机系
    Z, X = pts[:, :, 2], pts[:, :, 0]
    m = (disp != 0) & np.isfinite(pts).all(axis=2) & (Z > 0)
    if not object_paths:
        m &= (Z > 0.15) & (Z < 3.0) & (np.abs(X) < 1.2 * Z)
    colors = cv2.cvtColor(rl, cv2.COLOR_BGR2RGB)
    Pp, Cc = pts[m], colors[m]
    if len(Pp) == 0:
        raise SystemExit('这一对视差几乎全空，换一对试试')
    print(f'深度 1%~99%: {np.percentile(Z[m],1)*1000:.0f}~{np.percentile(Z[m],99)*1000:.0f} mm, '
          f'点云 {len(Pp)} 点')

    os.makedirs(OUT, exist_ok=True)
    ply = os.path.join(OUT, f'pointcloud_pair{suffix}.ply')
    with open(ply, 'w', encoding='ascii') as fp:
        fp.write('ply\nformat ascii 1.0\n')
        fp.write(f'element vertex {len(Pp)}\n')
        fp.write('property float x\nproperty float y\nproperty float z\n')
        fp.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for (Xc, Yc, Zc), (r, g, b) in zip(Pp, Cc):
            fp.write(f'{Xc:.5f} {Yc:.5f} {Zc:.5f} {r} {g} {b}\n')

    raw_valid = image_coordinates(matched)
    raw_px = np.where(raw_valid, disparity_sign * image_coordinates(raw_disparity), np.nan)
    np.savez_compressed(os.path.join(OUT, f'disparity_{suffix}.npz'),
                        raw_px=raw_px, filtered_px=np.where(disp != 0, disp, np.nan),
                        repaired=repaired, Q=Q)

    cv2.imwrite(os.path.join(OUT, f'rect_left_{suffix}.png'), rl)
    cv2.imwrite(os.path.join(OUT, f'rect_right_{suffix}.png'), rr)
    dvv = np.zeros((out_h, out_w, 3), np.uint8)
    magnitude = np.abs(disp)
    if magnitude.max() > 0:
        dn = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)
        dvv = cv2.applyColorMap(dn, cv2.COLORMAP_JET)
        dvv[disp == 0] = 0
    cv2.imwrite(os.path.join(OUT, f'disp_{suffix}.png'), dvv)

    def render(Pc, Ccol, yaw=20, pitch=10, W=800, H=600, zoom=1.0, source_pixels=None):
        c = Pc.mean(axis=0); P0 = Pc - c
        r = np.percentile(np.linalg.norm(P0, axis=1), 90)
        ya, pa = np.radians(yaw), np.radians(pitch)
        pos = np.array([np.sin(ya)*np.cos(pa), -np.sin(pa), np.cos(ya)*np.cos(pa)]) * 2.3 * r
        zax = -pos/np.linalg.norm(pos)
        xax = np.cross(np.array([0.0, -1.0, 0.0]), zax); xax /= np.linalg.norm(xax)
        yax = np.cross(zax, xax)
        Qc = ((np.stack([xax, yax, zax]) @ (P0 - pos).T).T)
        ok = Qc[:, 2] > 1e-3
        Qc, Ci = Qc[ok], Ccol[ok]
        zc = Qc[:, 2]
        f2 = zoom * 0.9 * max(W, H)
        u = f2 * Qc[:, 0] / zc + W / 2
        # The view axis points up, but image rows increase downward.
        v = H / 2 - f2 * Qc[:, 1] / zc
        img = np.full((H, W, 3), 255, np.uint8)
        visible = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        order = np.argsort(-zc[visible])
        ui = u[visible].astype(np.int32)[order]
        vi = v[visible].astype(np.int32)[order]
        img[vi, ui] = Ci[visible][order]
        preview = cv2.erode(img, np.ones((2, 2), np.uint8))
        if source_pixels is None:
            return preview
        pixel_map = np.full((H, W), -1, dtype=np.intp)
        pixel_map[vi, ui] = source_pixels[ok][visible][order]
        return preview, pixel_map

    front_view = render(Pp, Cc[:, ::-1], 20, 10)
    side_view = render(Pp, Cc[:, ::-1], -25, 30)
    cv2.imwrite(os.path.join(OUT, f'cloud_view1_{suffix}.png'), front_view)
    cv2.imwrite(os.path.join(OUT, f'cloud_view2_{suffix}.png'), side_view)
    cv2.imwrite(os.path.join(OUT, f'pair_{suffix}_overview.png'),
                np.hstack([cv2.resize(rl, (out_w//2, out_h//2)),
                           cv2.resize(rr, (out_w//2, out_h//2)),
                           cv2.resize(dvv, (out_w//2, out_h//2))]))
    print('已保存到', OUT)

    if not args.no_gui:
        source_pixels = np.where(repaired[m], -1, np.flatnonzero(m))

        def draw_cloud(yaw, pitch, zoom):
            return render(Pp, Cc[:, ::-1], yaw, pitch, zoom=zoom,
                          source_pixels=source_pixels)

        measure_points(rl, pts, m & ~repaired, PAIR, draw_cloud)


def nearest_pixel(valid, x, y, radius):
    height, width = valid.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    if valid[y, x]:
        return x, y
    left, top = max(0, x - radius), max(0, y - radius)
    rows, cols = np.nonzero(valid[top:y + radius + 1, left:x + radius + 1])
    if not rows.size:
        return None
    distance_sq = (cols + left - x) ** 2 + (rows + top - y) ** 2
    closest = int(np.argmin(distance_sq))
    if distance_sq[closest] > radius ** 2:
        return None
    return int(cols[closest] + left), int(rows[closest] + top)


def measure_points(image, points, valid, pair, draw_cloud):
    window = f'Measure {pair}.jpg - Left click | D: depth overlay | C: clear | Q: quit'
    clicks = []
    status = 'Click two points on the object'

    dirty = True
    show_depth = True
    depth_preview = (image * 0.35).astype(np.uint8)
    depth_preview[valid] = image[valid]

    def add_point(x, y):
        nonlocal status, dirty
        dirty = True
        if len(clicks) == 2:
            clicks.clear()
        clicks.append((x, y))
        status = 'Click the second point'
        if len(clicks) == 2:
            (x1, y1), (x2, y2) = clicks
            distance_mm = 1000 * np.linalg.norm(points[y2, x2] - points[y1, x1])
            status = f'Distance: {distance_mm:.1f} mm'
            print(f'两点距离 ({x1}, {y1}) 到 ({x2}, {y2}): {distance_mm:.1f} mm', flush=True)

    def on_mouse(event, x, y, flags, param):
        nonlocal status, dirty
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = nearest_pixel(valid, x, y, radius=8)
        if point is None:
            status = 'No depth nearby. Choose a bright area.'
            dirty = True
        else:
            add_point(*point)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, 540, 720)
    cv2.moveWindow(window, 30, 40)
    cv2.setMouseCallback(window, on_mouse)
    cloud_window = f'3D point cloud {pair}.ply | Left drag: rotate | Wheel: zoom | Right click: measure'
    yaw, pitch, zoom = 20.0, 10.0, 1.0
    anchor = None
    redraw = False
    cloud_preview, pixel_map = draw_cloud(yaw, pitch, zoom)

    def on_cloud_mouse(event, x, y, flags, param):
        nonlocal yaw, pitch, zoom, anchor, redraw, status, dirty
        if event == cv2.EVENT_RBUTTONDOWN:
            point = nearest_pixel(pixel_map >= 0, x, y, radius=8)
            if point is None:
                status = 'No point nearby. Right-click on the cloud.'
                dirty = True
            else:
                u, v = point
                row, col = divmod(int(pixel_map[v, u]), valid.shape[1])
                add_point(col, row)
        elif event == cv2.EVENT_LBUTTONDOWN:
            anchor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            anchor = None
        elif event == cv2.EVENT_MOUSEMOVE and anchor is not None:
            if not flags & cv2.EVENT_FLAG_LBUTTON:
                anchor = None
                return
            yaw += (x - anchor[0]) * 0.4
            pitch = float(np.clip(pitch + (y - anchor[1]) * 0.4, -85, 85))
            anchor = (x, y)
            redraw = True
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16)
            zoom = float(np.clip(zoom * (1.15 if delta > 0 else 1 / 1.15), 0.25, 5))
            redraw = True
    try:
        cv2.namedWindow(cloud_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(cloud_window, 640, 480)
        cv2.moveWindow(cloud_window, 590, 40)
        cv2.imshow(cloud_window, cloud_preview)
        cv2.setMouseCallback(cloud_window, on_cloud_mouse)
        while True:
            if redraw:
                cloud_preview, pixel_map = draw_cloud(yaw, pitch, zoom)
                redraw = False
                dirty = True
            if dirty:
                preview = (depth_preview if show_depth else image).copy()
                if len(clicks) == 2:
                    cv2.line(preview, clicks[0], clicks[1], (0, 255, 255), 2)
                for index, point in enumerate(clicks, 1):
                    cv2.drawMarker(preview, point, (0, 0, 255), cv2.MARKER_CROSS, 17, 2)
                    cv2.putText(preview, f'P{index}', (point[0] + 9, point[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(preview, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.imshow(window, preview)
                if cv2.getWindowProperty(cloud_window, cv2.WND_PROP_VISIBLE) >= 1:
                    marked_cloud = cloud_preview.copy()
                    cv2.putText(marked_cloud, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (30, 30, 220), 2)
                    cv2.imshow(cloud_window, marked_cloud)
                dirty = False
            pressed = cv2.waitKey(30) & 0xFF
            if pressed in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if pressed == ord('d'):
                show_depth = not show_depth
                dirty = True
            if pressed == ord('c'):
                dirty = True
                clicks.clear()
                status = 'Click two points on the object'
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
