r"""批量跑所有可用图对：逐对校正 -> SGBM 视差 -> 统计有效匹配率 -> 给最好的几对出点云。

用法:  python batch_pairs.py [校正宽度] [取最好的对数]
输出:  results\batch\ 下的 report.md / batch_report.json / 每对预览 / best_*.ply / contact_sheet.png
"""
import cv2, numpy as np, glob, os, sys, json, csv

HERE = os.path.dirname(os.path.abspath(__file__))
LEFT_DIR = os.path.join(HERE, 'pairs', 'left')
RIGHT_DIR = os.path.join(HERE, 'pairs', 'right')
OUT = os.path.join(HERE, 'results')
BATCH = os.path.join(OUT, 'batch')
CACHE = os.path.join(OUT, 'corners.npz')
PATTERN = (9, 6)
SQUARE = 0.05
OUT_W = int(sys.argv[1]) if len(sys.argv) > 1 else 960
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SHOW_RESULT = True      # 跑完弹窗显示对比图
os.makedirs(BATCH, exist_ok=True)
# 清掉上一次的产物，避免不同图对号的旧文件混进拼图
for _old in (glob.glob(os.path.join(BATCH, 'rank*.png'))
             + glob.glob(os.path.join(BATCH, 'best*.ply'))
             + glob.glob(os.path.join(BATCH, 'contact_sheet.png'))):
    try:
        os.remove(_old)
    except OSError:
        pass


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def key(p):
    return int(''.join(c for c in os.path.basename(p) if c.isdigit()))


def render(Pp, Cc, yaw=20, pitch=10, W=420, H=320):
    c = Pp.mean(axis=0)
    P0 = Pp - c
    r = np.percentile(np.linalg.norm(P0, axis=1), 90)
    if r <= 0:
        return np.full((H, W, 3), 255, np.uint8)
    ya, pa = np.radians(yaw), np.radians(pitch)
    pos = np.array([np.sin(ya) * np.cos(pa), -np.sin(pa), np.cos(ya) * np.cos(pa)]) * 2.3 * r
    zax = -pos / np.linalg.norm(pos)
    xax = np.cross(np.array([0.0, -1.0, 0.0]), zax)
    xax /= (np.linalg.norm(xax) + 1e-12)
    yax = np.cross(zax, xax)
    Qc = (np.stack([xax, yax, zax]) @ (P0 - pos).T).T
    ok = Qc[:, 2] > 1e-3
    Qc, Ci = Qc[ok], Cc[ok]
    zc = Qc[:, 2]
    f2 = 0.9 * max(W, H)
    u = f2 * Qc[:, 0] / zc + W / 2
    v = f2 * Qc[:, 1] / zc + H / 2
    img = np.full((H, W, 3), 255, np.uint8)
    order = np.argsort(-zc)
    ui = np.clip(u, 0, W - 1).astype(np.int32)[order]
    vi = np.clip(v, 0, H - 1).astype(np.int32)[order]
    img[vi, ui] = Ci[order]
    return cv2.dilate(img, np.ones((2, 2), np.uint8))


def main():
    z = np.load(CACHE, allow_pickle=True)
    cL, cR, used = z['cL'], z['cR'], z['used']
    sizeL = tuple(int(v) for v in z['sizeL']); sizeR = tuple(int(v) for v in z['sizeR'])
    inv = os.path.join(OUT, 'stereo_calib_result_pairs.npz')
    inl = np.ones(len(used), bool)
    if os.path.exists(inv):
        t = np.load(inv, allow_pickle=True)
        if 'T_pairs' in t.files and 'inlier_mask' in t.files:
            tp = np.array(t['T_pairs'], dtype=float)
            bl = np.linalg.norm(tp, axis=1)
            cv_ = float(np.std(bl) / max(np.mean(bl), 1e-9))
            if cv_ < 0.05:
                inl = t['inlier_mask'].astype(bool)
                print(f'基线一致性 CV={cv_:.3f}，按标定内点集过滤')
            else:
                print(f'基线波动 CV={cv_:.3f}（手持拍摄属正常），不按基线剔除，全部 {len(used)} 对都跑')

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

    out_w, out_h = min(OUT_W, sizeL[0]), int(round(min(OUT_W, sizeL[0]) * sizeL[1] / sizeL[0]))
    f = 0.98 * K2[0, 0] * out_w / sizeR[0]
    cx, cy = out_w / 2.0, out_h / 2.0
    bs = 9
    sgbm_base = dict(minDisparity=0, blockSize=bs,
                     P1=8 * 3 * bs ** 2, P2=32 * 3 * bs ** 2, disp12MaxDiff=2,
                     uniquenessRatio=5, speckleWindowSize=300, speckleRange=2,
                     preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    rows = []
    for i, pnum in enumerate(used):
        if not inl[i]:
            print(f'[{pnum:02d}] 跳过（标定异常对）')
            continue
        k = i
        Rl, _ = cv2.Rodrigues(rv1[k]); Rr, _ = cv2.Rodrigues(rv2[k])
        R = Rr @ Rl.T
        T = (tv2[k] - R @ tv1[k]).ravel()
        if float(np.linalg.norm(T)) > 2.0 * np.median([np.linalg.norm(tv2[j] - (cv2.Rodrigues(rv2[j])[0] @ cv2.Rodrigues(rv1[j])[0].T) @ tv1[j])
                                                       for j in range(len(used))]):
            print(f'[{pnum:02d}] 基线离群，跳过')
            continue
        nd_req = float(f) * float(np.linalg.norm(T)) / 0.5
        if (not np.isfinite(nd_req)) or nd_req <= 0:
            print('视差范围估算异常，按 400 处理')
            nd_req = 400.0
        nd = 16 * (int(nd_req) // 16 + 1)
        if nd > 512:
            nd = 512
        sgbm_t = dict(sgbm_base, numDisparities=nd)
        R1, R2 = cv2.stereoRectify(K1, D1, K2, D2, sizeL, R, T,
                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)[0:2]
        P1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
        P2 = np.array([[f, 0, cx, f * T[0]], [0, f, cy, 0], [0, 0, 1, 0]], np.float64)
        Q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1.0 / T[0], 0]], np.float64)

        u1 = cv2.undistortPoints(cl[k], K1, D1, R=R1, P=P1).reshape(-1, 2)
        u2 = cv2.undistortPoints(cr[k], K2, D2, R=R2, P=P2).reshape(-1, 2)
        dy = float(np.abs(u1[:, 1] - u2[:, 1]).mean())
        db = float(np.mean(u1[:, 0] - u2[:, 0]))
        board_depth = abs(f * T[0]) / max(db, 1e-6)
        bx, by, bw, bh = cv2.boundingRect(u1.astype(np.int32))
        bmask = np.zeros((out_h, out_w), np.uint8)
        bmask[max(0, by):min(out_h, by + bh), max(0, bx):min(out_w, bx + bw)] = 1

        iml = imread_u(Ls[pnum - 1])
        imr = imread_u(Rs[pnum - 1])
        if (imr.shape[1], imr.shape[0]) != sizeR:
            imr = cv2.resize(imr, sizeR, interpolation=cv2.INTER_AREA)
        m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (out_w, out_h), cv2.CV_32FC1)
        m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (out_w, out_h), cv2.CV_32FC1)
        rl = cv2.remap(iml, m1[0], m1[1], cv2.INTER_LINEAR)
        rr = cv2.remap(imr, m2[0], m2[1], cv2.INTER_LINEAR)
        gl = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
        try:
            disp = cv2.StereoSGBM_create(**sgbm_t).compute(gl, gr).astype(np.float32) / 16.0
        except cv2.error as e:
            print(f'[{pnum:02d}] SGBM 失败，跳过: {str(e).splitlines()[-1][:80]}')
            continue
        valid = (disp > 0) & (disp < nd - 1)
        va = float(valid.mean())
        vb = float((valid & (bmask > 0)).sum() / max((bmask > 0).sum(), 1))
        fnum = int(''.join(c for c in os.path.basename(Ls[pnum - 1]) if c.isdigit()))
        print(f'[{fnum:02d}] 基线 {np.linalg.norm(T)*1000:6.1f}mm  极线 {dy:.2f}px  '
              f'棋盘深度 {board_depth*1000:5.0f}mm  有效 {va*100:4.1f}% (棋盘区 {vb*100:4.1f}%)')
        rows.append(dict(pair=int(pnum), file=int(fnum), baseline_mm=float(np.linalg.norm(T) * 1000),
                         epipolar_px=dy, board_depth_mm=float(board_depth * 1000),
                         valid_all=va, valid_board=vb))

    # 排名以“棋盘区域有效匹配率”为准（棋盘是真实目标物，比全图更有意义）
    rows.sort(key=lambda d: -d['valid_board'])
    with open(os.path.join(BATCH, 'batch_report.json'), 'w', encoding='utf-8') as fp:
        json.dump(rows, fp, indent=2, ensure_ascii=False)
    with open(os.path.join(BATCH, 'batch_report.csv'), 'w', newline='', encoding='utf-8-sig') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    md = ['# 逐对匹配质量（%d 对，方格 %.0f mm，校正宽度 %d）' % (len(rows), SQUARE * 1000, out_w), '',
          '| 排名 | 图对 | 基线 mm | 极线残差 px | 棋盘深度 mm | 整体有效 % | 棋盘区有效 % |',
          '|---|---|---|---|---|---|---|']
    for r_i, d in enumerate(rows, 1):
        md.append('| %d | %d | %.1f | %.2f | %.0f | %.1f | %.1f |' % (
            r_i, d.get('file', d['pair']), d['baseline_mm'], d['epipolar_px'], d['board_depth_mm'],
            d['valid_all'] * 100, d['valid_board'] * 100))
    with open(os.path.join(BATCH, 'report.md'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(md) + '\n')

    # ---- 最好的 TOPN 对出点云 + 预览 ----
    tiles = []
    for rank, d in enumerate(rows[:TOPN], 1):
        pnum = d['pair']; k = used.tolist().index(pnum)
        Rl, _ = cv2.Rodrigues(rv1[k]); Rr, _ = cv2.Rodrigues(rv2[k])
        R = Rr @ Rl.T
        T = (tv2[k] - R @ tv1[k]).ravel()
        nd_req = float(f) * float(np.linalg.norm(T)) / 0.5
        if (not np.isfinite(nd_req)) or nd_req <= 0:
            print('视差范围估算异常，按 400 处理')
            nd_req = 400.0
        nd = 16 * (int(nd_req) // 16 + 1)
        if nd > 512:
            nd = 512
        sgbm_t = dict(sgbm_base, numDisparities=nd)
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
        pts = cv2.reprojectImageTo3D(disp, Q)
        Z, X = pts[:, :, 2], pts[:, :, 0]
        m = (disp > 0) & np.isfinite(Z) & (Z > 0.2) & (Z < 4.0) & (np.abs(X) < 1.2 * Z)
        colors = cv2.cvtColor(rl, cv2.COLOR_BGR2RGB)
        Pp, Cc = pts[m], colors[m]
        if len(Pp) < 1000:
            print(f'[{pnum:02d}] 点太少，跳过点云'); continue
        ply = os.path.join(BATCH, f'best{rank:02d}_pair{pnum:02d}.ply')
        with open(ply, 'w', encoding='ascii') as fp:
            fp.write('ply\nformat ascii 1.0\n')
            fp.write(f'element vertex {len(Pp)}\n')
            fp.write('property float x\nproperty float y\nproperty float z\n')
            fp.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
            for (Xc, Yc, Zc), (r_, g_, b_) in zip(Pp, Cc):
                fp.write(f'{Xc:.5f} {Yc:.5f} {Zc:.5f} {r_} {g_} {b_}\n')
        dv = np.zeros((out_h, out_w, 3), np.uint8)
        if disp.max() > 0:
            dn = np.clip(disp / disp.max() * 255, 0, 255).astype(np.uint8)
            dv = cv2.applyColorMap(dn, cv2.COLORMAP_JET)
            dv[disp <= 0] = 0
        view = render(Pp, Cc)
        tile = np.hstack([cv2.resize(rl, (420, 560)), cv2.resize(rr, (420, 560)),
                          cv2.resize(dv, (420, 560)),
                          cv2.copyMakeBorder(view, (560 - view.shape[0]) // 2, 560 - view.shape[0] - (560 - view.shape[0]) // 2, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))])
        cv2.putText(tile, f'pair {pnum}  valid {d["valid_all"]*100:.0f}%  depth {d["board_depth_mm"]:.0f}mm',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(BATCH, f'rank{rank:02d}_pair{pnum:02d}.png'), tile)
        tiles.append(cv2.resize(tile, (tile.shape[1] // 2, tile.shape[0] // 2)))
        print(f'  -> rank {rank}: pair {pnum}, {len(Pp)} 点, {ply}')

    if tiles:
        cv2.imwrite(os.path.join(BATCH, 'contact_sheet.png'), np.vstack(tiles))
    print('\n完成，结果在', BATCH)

    if SHOW_RESULT:
        from show_util import show_images
        show_images([os.path.join(BATCH, 'contact_sheet.png')],
                    title='前 %d 对结果（按任意键关闭）' % TOPN)


if __name__ == '__main__':
    main()
