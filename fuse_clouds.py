r"""多视角点云融合：把若干对照片的点云统一变换到「棋盘格坐标系」后合并，得到比单对完整的场景点云。

原理：calibrateCamera 对每张左图都能给出棋盘格在左相机坐标系下的位姿 (rv1, tv1)，
      p_cam = R_b @ p_board + t_b  =>  p_board = R_b^T @ (p_cam - t_b)
      所有图对都换到同一块棋盘的坐标系里，就自动对齐、可以叠加。

用法:  python fuse_clouds.py [校正宽度] [格子毫米(降采样体素)] [图对号...]
       不给图对号时，默认取 results\batch\batch_report.json 里棋盘区匹配率最高的 6 对。
输出:  results\fused\fused_pointcloud.ply  results\fused\fused_view1.png / view2.png
"""
import cv2, numpy as np, glob, os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
LEFT_DIR = os.path.join(HERE, 'pairs', 'left')
RIGHT_DIR = os.path.join(HERE, 'pairs', 'right')
OUT = os.path.join(HERE, 'results')
FUSED = os.path.join(OUT, 'fused')
CACHE = os.path.join(OUT, 'corners.npz')
PATTERN = (9, 6)
SQUARE = 0.05
OUT_W = int(sys.argv[1]) if len(sys.argv) > 1 else 1440
VOXEL_MM = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SHOW_RESULT = True      # 跑完弹窗显示三个视角
PICK = [int(v) for v in sys.argv[3:]] if len(sys.argv) > 3 else None
os.makedirs(FUSED, exist_ok=True)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def key(p):
    return int(''.join(c for c in os.path.basename(p) if c.isdigit()))


def render(Pp, Cc, yaw=20, pitch=10, W=1000, H=750):
    c = Pp.mean(axis=0)
    P0 = Pp - c
    r = np.percentile(np.linalg.norm(P0, axis=1), 90)
    ya, pa = np.radians(yaw), np.radians(pitch)
    pos = np.array([np.sin(ya) * np.cos(pa), -np.sin(pa), np.cos(ya) * np.cos(pa)]) * 2.3 * r
    zax = -pos / np.linalg.norm(pos)
    xax = np.cross(np.array([0.0, -1.0, 0.0]), zax); xax /= np.linalg.norm(xax)
    yax = np.cross(zax, xax)
    Qc = (np.stack([xax, yax, zax]) @ (P0 - pos).T).T
    ok = Qc[:, 2] > 1e-3
    Qc, Ci = Qc[ok], Cc[ok]
    zc = Qc[:, 2]
    f2 = 0.9 * max(W, H)
    u = f2 * Qc[:, 0] / zc + W / 2
    v = f2 * Qc[:, 1] / zc + H / 2
    img = np.full((H, W, 3), 18, np.uint8)
    order = np.argsort(-zc)
    ui = np.clip(u, 0, W - 1).astype(np.int32)[order]
    vi = np.clip(v, 0, H - 1).astype(np.int32)[order]
    img[vi, ui] = Ci[order]
    return cv2.dilate(img, np.ones((3, 3), np.uint8))


def main():
    z = np.load(CACHE, allow_pickle=True)
    cL, cR, used = z['cL'], z['cR'], z['used']
    sizeL = tuple(int(v) for v in z['sizeL']); sizeR = tuple(int(v) for v in z['sizeR'])
    Ls = sorted(glob.glob(os.path.join(LEFT_DIR, '*.jpg')), key=key)
    Rs = sorted(glob.glob(os.path.join(RIGHT_DIR, '*.jpg')), key=key)
    def _sig(d):
        fs = sorted(glob.glob(os.path.join(d, '*.jpg')), key=key)
        return '|'.join(f'{os.path.basename(p)}:{os.path.getsize(p)}' for p in fs)
    if 'sig' in z.files and str(z['sig']) != (_sig(LEFT_DIR) + '#' + _sig(RIGHT_DIR)):
        raise SystemExit('pairs 里的照片和 results\\corners.npz 不一致，请先重跑 stereo_calib_pairs.py')

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE
    ol = [objp] * len(used)
    cl = [c.reshape(-1, 1, 2).astype(np.float32) for c in cL]
    cr = [c.reshape(-1, 1, 2).astype(np.float32) for c in cR]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    _, K1, D1, rv1, tv1 = cv2.calibrateCamera(ol, cl, sizeL, None, None, criteria=crit)
    _, K2, D2, rv2, tv2 = cv2.calibrateCamera(ol, cr, sizeR, None, None, criteria=crit)

    pick = list(PICK) if PICK else None
    if pick is None:
        rep = os.path.join(OUT, 'batch', 'batch_report.json')
        if os.path.exists(rep):
            rows = json.load(open(rep, encoding='utf-8'))
            pick = [int(r['pair']) for r in rows[:6]]
        else:
            pick = [13, 12, 23, 8, 28, 29]
    _avail = set(int(v) for v in used.tolist())
    _missing = [v for v in pick if v not in _avail]
    pick = [v for v in pick if v in _avail]
    if _missing:
        print('跳过没有检测到棋盘格的图对:', _missing)
    if not pick:
        pick = [int(v) for v in used.tolist()]
        print('没有有效图对列表（默认列表里的编号都不存在），改用全部 %d 对' % len(pick))
    print('融合图对(%d 对):' % len(pick), pick)

    out_w = min(OUT_W, sizeL[0])
    out_h = int(round(out_w * sizeL[1] / sizeL[0]))
    f = 0.98 * K2[0, 0] * out_w / sizeR[0]
    cx, cy = out_w / 2.0, out_h / 2.0
    # 视差搜索范围要覆盖最近 ~0.5 m：d_max = f * B / Zmin
    bl_cand = []
    for p in pick:
        kk = used.tolist().index(p)
        Rl0, _ = cv2.Rodrigues(rv1[kk]); Rr0, _ = cv2.Rodrigues(rv2[kk])
        R0 = Rr0 @ Rl0.T
        bl_cand.append(float(np.linalg.norm(tv2[kk] - R0 @ tv1[kk])))
    bl_med = float(np.median(bl_cand)) if bl_cand else 0.4
    if (not np.isfinite(bl_med)) or bl_med <= 0:
        print('基线无效(%.3f)，按 0.4 m 估算视差范围' % bl_med)
        bl_med = 0.4
    nd_req = float(f) * float(bl_med) / 0.5
    if (not np.isfinite(nd_req)) or nd_req <= 0:
        print('视差范围估算异常(f=%r, 基线=%r)，按 400 处理' % (f, bl_med))
        nd_req = 400.0
    nd = 16 * (int(nd_req) // 16 + 1)
    nd = 512 if nd > 512 else nd
    bs = 9
    sgbm_t = dict(minDisparity=0, numDisparities=nd, blockSize=bs,
                  P1=8 * 3 * bs ** 2, P2=32 * 3 * bs ** 2, disp12MaxDiff=2,
                  uniquenessRatio=5, speckleWindowSize=300, speckleRange=2,
                  preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    print(f'校正宽度 {out_w}x{out_h}, f={f:.0f}, 视差范围 {nd}')

    allP, allC = [], []
    for pnum in pick:
        k = used.tolist().index(pnum)
        Rl, _ = cv2.Rodrigues(rv1[k]); Rr, _ = cv2.Rodrigues(rv2[k])
        R = Rr @ Rl.T
        T = (tv2[k] - R @ tv1[k]).ravel()
        if float(np.linalg.norm(T)) > 2.0 * bl_med:
            print(f'  pair {pnum:02d}: 基线离群 {np.linalg.norm(T)*1000:.0f}mm，跳过')
            continue
        R1, R2 = cv2.stereoRectify(K1, D1, K2, D2, sizeL, R, T,
                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)[0:2]
        P1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
        P2 = np.array([[f, 0, cx, f * T[0]], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
        Q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1.0 / T[0], 0]], np.float64)

        iml = imread_u(Ls[pnum - 1]); imr = imread_u(Rs[pnum - 1])
        if (imr.shape[1], imr.shape[0]) != sizeR:
            imr = cv2.resize(imr, sizeR, interpolation=cv2.INTER_AREA)
        m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (out_w, out_h), cv2.CV_32FC1)
        m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (out_w, out_h), cv2.CV_32FC1)
        rl = cv2.remap(iml, m1[0], m1[1], cv2.INTER_LINEAR)
        rr = cv2.remap(imr, m2[0], m2[1], cv2.INTER_LINEAR)
        gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY); gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
        disp = cv2.StereoSGBM_create(**sgbm_t).compute(gl, gr).astype(np.float32) / 16.0
        disp[(disp <= 0) | (disp >= nd - 1)] = 0
        disp = cv2.medianBlur(disp, 5)
        pts_rect = cv2.reprojectImageTo3D(disp, Q)   # 校正后的虚拟相机坐标系
        # 旋转回原始左相机坐标系: p_cam = R1^T @ p_rect  =>  (p_rect^T R1)
        with np.errstate(invalid='ignore', divide='ignore'):
            pts = (pts_rect.reshape(-1, 3) @ R1).reshape(pts_rect.shape)
        Z, X, Y = pts[:, :, 2], pts[:, :, 0], pts[:, :, 1]
        # 只保留棋盘格附近的一段景深（0.35~1.5 m），甩掉远处墙面
        kk = SQUARE / 0.025          # 尺度系数：深度窗口随方格尺寸一起缩放
        m = (disp > 0) & np.isfinite(Z) & (Z > 0.35 * kk) & (Z < 1.5 * kk) & (np.abs(X) < 0.9 * Z) & (np.abs(Y) < 0.9 * Z)
        Pc = pts[m]                                  # (N,3) 相机系
        Cc = cv2.cvtColor(rl, cv2.COLOR_BGR2RGB)[m]

        # 左相机坐标系 -> 棋盘格坐标系
        Rb, _ = cv2.Rodrigues(rv1[k])
        tb = tv1[k].ravel()
        Pb = (Pc - tb) @ Rb                        # (Rb^T (p - t))^T
        # 棋盘格坐标系里再裁一刀：目标物都在棋盘前后 0.9 m、左右上下 0.8 m 内
        keep = (np.abs(Pb[:, 0]) < 0.6 * kk) & (np.abs(Pb[:, 1]) < 0.6 * kk) & (np.abs(Pb[:, 2]) < 0.8 * kk)
        Pb, Cc = Pb[keep], Cc[keep]
        allP.append(Pb); allC.append(Cc)
        print(f'  pair {pnum:02d}: {len(Pb)} 点')

    P = np.vstack(allP); C = np.vstack(allC)
    print(f'合并原始 {len(P)} 点')
    # 体素降采样
    vox = VOXEL_MM / 1000.0
    q = np.floor(P / vox).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    P, C = P[idx], C[idx]
    print(f'体素降采样({VOXEL_MM}mm)后 {len(P)} 点')

    ply = os.path.join(FUSED, 'fused_pointcloud.ply')
    with open(ply, 'w', encoding='ascii') as fp:
        fp.write('ply\nformat ascii 1.0\n')
        fp.write(f'element vertex {len(P)}\n')
        fp.write('property float x\nproperty float y\nproperty float z\n')
        fp.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for (X_, Y_, Z_), (r_, g_, b_) in zip(P, C):
            fp.write(f'{X_:.5f} {Y_:.5f} {Z_:.5f} {r_} {g_} {b_}')
            fp.write('\n')
    print('已保存', ply)
    ext = (P.max(axis=0) - P.min(axis=0)) * 1000
    print('点云包围盒(mm): %.0f x %.0f x %.0f' % tuple(ext))
    cv2.imwrite(os.path.join(FUSED, 'fused_view1.png'), render(P, C, 15, 8))
    cv2.imwrite(os.path.join(FUSED, 'fused_view2.png'), render(P, C, -30, 25))
    cv2.imwrite(os.path.join(FUSED, 'fused_view3.png'), render(P, C, 0, 0))
    print('预览图已保存到', FUSED)

    if SHOW_RESULT:
        from show_util import show_images
        show_images([os.path.join(FUSED, 'fused_view1.png'),
                     os.path.join(FUSED, 'fused_view2.png'),
                     os.path.join(FUSED, 'fused_view3.png')],
                    title='融合点云（按任意键关闭）')


if __name__ == '__main__':
    main()
