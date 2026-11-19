r"""双目标定：左右两台相机分别拍同一块棋盘格，做立体标定。

数据放置(默认)：
    <本脚本所在目录>\pairs\left\*.jpg   左相机照片
    <本脚本所在目录>\pairs\right\*.jpg  右相机照片（文件名数字需与左边对应）

也可以在命令行给出两个目录：
    python stereo_calib_pairs.py <左目录> <右目录>

输出： results\stereo_calib_result_pairs.npz, results\calib_report.json
      npz 内含 mtx_l dist_l mtx_r dist_r R T E F R1 R2 P1 P2 Q roi1 roi2 等，
      与你原来的 calibrate_new.py 输出键名一致。

注意：两台相机若没有刚性固定、也不是同一时刻触发，本脚本会用逐对棋盘位姿筛掉
      相对姿态异常的图对，并把“逐对相对位姿”一并存进 npz（R_pairs/T_pairs），
      供 run_pair.py 做逐对校正使用。
"""
import cv2, numpy as np, glob, os, json, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LEFT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'pairs', 'left')
RIGHT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'pairs', 'right')
OUT = os.path.join(HERE, 'results')
CACHE = os.path.join(OUT, 'corners.npz')
PATTERN = (9, 6)      # 内角点个数 (列, 行)
SQUARE = 0.05         # 方格真实边长，单位 m（板宽 500 mm / 10 格）
CANON_R = None        # 右相机统一分辨率；None = 用第一张的原始尺寸
SHOW_EACH = True        # 检测角点时逐对弹窗（和最早的 calibrate.py 一样）
SHOW_DELAY_MS = 300     # 每对停留时间(ms)；设 0 表示按键才继续
FORCE_REDETECT = False  # True = 忽略角点缓存，全部重新检测（这样才会逐对弹窗）
os.makedirs(OUT, exist_ok=True)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def key(p):
    return int(''.join(c for c in os.path.basename(p) if c.isdigit()))


def detect(gray, pat):
    """大图(宽>3000)先半分辨率粗检、再回到原图 cornerSubPix 精修，速度快一个数量级。"""
    fscale = 0.5 if gray.shape[1] > 3000 else 1.0
    g = gray if fscale == 1.0 else cv2.resize(gray, None, fx=fscale, fy=fscale, interpolation=cv2.INTER_AREA)
    try:
        ok, c = cv2.findChessboardCornersSB(g, pat, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok and c is not None and len(c) == pat[0] * pat[1]:
            pts = c.reshape(-1, 1, 2).astype(np.float32)
            if fscale != 1.0:
                pts = pts / fscale
                pts = cv2.cornerSubPix(gray, pts, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
            return pts
    except cv2.error:
        pass
    ok, c = cv2.findChessboardCorners(g, pat, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok and c is not None:
        pts = c.astype(np.float32)
        if fscale != 1.0:
            pts = cv2.cornerSubPix(gray, pts * fscale, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
        return pts
    return None


def show_pair(iml, imr, cl, cr, idx, delay=None):
    """把左右两张（画好角点）并排弹窗显示，和最早那版 calibrate.py 的体验一样。"""
    global SHOW_EACH
    if not SHOW_EACH:
        return
    box = 620

    def prep(im, c):
        h, w = im.shape[:2]
        sc = box / float(w)
        d = cv2.resize(im, (box, int(round(h * sc))))
        if c is not None:
            cc = (c.reshape(-1, 1, 2).astype(np.float64) * sc).astype(np.float32)
            cv2.drawChessboardCorners(d, PATTERN, cc, True)
            cv2.putText(d, 'OK', (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(d, 'FAIL', (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return d

    a = prep(iml, cl)
    b = prep(imr, cr)
    h = max(a.shape[0], b.shape[0])
    pad = lambda x: cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                       value=(30, 30, 30))
    canvas = np.hstack([pad(a), pad(b)])
    cv2.putText(canvas, 'pair %02d  (q=close windows)' % idx, (8, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.namedWindow('检测中', cv2.WINDOW_NORMAL)
    cv2.imshow('检测中', canvas)
    key = cv2.waitKey(SHOW_DELAY_MS if delay is None else delay)
    if key in (ord('q'), 27):
        SHOW_EACH = False
        cv2.destroyAllWindows()


Ls = sorted(glob.glob(os.path.join(LEFT_DIR, '*.jpg')), key=key)
Rs = sorted(glob.glob(os.path.join(RIGHT_DIR, '*.jpg')), key=key)
print(f'左图 {len(Ls)} 张, 右图 {len(Rs)} 张')
assert Ls and Rs, '没找到 jpg 图片'
N = min(len(Ls), len(Rs))

objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE

def sig_of(files):
    return '|'.join(f'{os.path.basename(p)}:{os.path.getsize(p)}' for p in files)


SIG = sig_of(Ls) + '#' + sig_of(Rs)

use_cache = False
if os.path.exists(CACHE) and not FORCE_REDETECT:
    z = np.load(CACHE, allow_pickle=True)
    if 'sig' in z.files and str(z['sig']) == SIG:
        cL, cR, used = z['cL'], z['cR'], z['used']
        sizeL, sizeR = tuple(int(v) for v in z['sizeL']), tuple(int(v) for v in z['sizeR'])
        use_cache = True
        print('使用已有角点缓存:', CACHE)

        # 即使命中缓存，也把每一对的“左+右”并排显示出来（和最早的 calibrate.py 一样）
        if SHOW_EACH:
            print('逐对显示左右配对（%d 对，按 q 关闭后续窗口）…' % len(used))
            for _i, _pnum in enumerate(used):
                _pos = int(_pnum) - 1
                _iml = imread_u(Ls[_pos])
                _imr = imread_u(Rs[_pos])
                if _iml is None or _imr is None:
                    continue
                if (_imr.shape[1], _imr.shape[0]) != sizeR:
                    _imr = cv2.resize(_imr, sizeR, interpolation=cv2.INTER_AREA)
                show_pair(_iml, _imr,
                          cL[_i].reshape(-1, 1, 2).astype(np.float32),
                          cR[_i].reshape(-1, 1, 2).astype(np.float32),
                          int(_pnum))
    else:
        print('照片与旧缓存不一致（换图了），重新检测角点…')

if not use_cache:
    cL, cR, used = [], [], []
    sizeL = sizeR = None
    t0 = time.time()
    for i in range(N):
        iml = imread_u(Ls[i])
        imr = imread_u(Rs[i])
        if iml is None or imr is None:
            print(f'[{i+1:02d}] 读取失败'); continue
        sizeL = (iml.shape[1], iml.shape[0])
        if sizeR is None:
            sizeR = (imr.shape[1], imr.shape[0]) if CANON_R is None else CANON_R
        if (imr.shape[1], imr.shape[0]) != sizeR:
            imr = cv2.resize(imr, sizeR, interpolation=cv2.INTER_AREA)
        cl = detect(cv2.cvtColor(iml, cv2.COLOR_BGR2GRAY), PATTERN)
        cr = detect(cv2.cvtColor(imr, cv2.COLOR_BGR2GRAY), PATTERN)
        print(f'[{i+1:02d}] 左 {sizeL[0]}x{sizeL[1]} {"OK" if cl is not None else "失败"} | '
              f'右 {sizeR[0]}x{sizeR[1]} {"OK" if cr is not None else "失败"}')
        if cl is not None and cr is not None:
            cL.append(cl.reshape(-1, 2)); cR.append(cr.reshape(-1, 2)); used.append(i + 1)
        show_pair(iml, imr, cl, cr, i + 1)
    cL = np.array(cL); cR = np.array(cR); used = np.array(used)
    np.savez(CACHE, cL=cL, cR=cR, used=used, sizeL=np.array(sizeL), sizeR=np.array(sizeR),
             sig=np.array(SIG))
    print(f'角点检测完成 {time.time()-t0:.0f}s，有效图对 {len(used)}/{N}')

if len(used) < 6:
    raise SystemExit('有效图对太少，无法标定')

sizeL = tuple(int(v) for v in sizeL); sizeR = tuple(int(v) for v in sizeR)
obj_list = [objp] * len(used)
cL_list = [c.reshape(-1, 1, 2).astype(np.float32) for c in cL]
cR_list = [c.reshape(-1, 1, 2).astype(np.float32) for c in cR]
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)

rmsL, K1, D1, rv1, tv1 = cv2.calibrateCamera(obj_list, cL_list, sizeL, None, None, criteria=crit)
rmsR, K2, D2, rv2, tv2 = cv2.calibrateCamera(obj_list, cR_list, sizeR, None, None, criteria=crit)
print(f'\n单目标定  左 RMS={rmsL:.3f}px fx={K1[0,0]:.1f}   右 RMS={rmsR:.3f}px fx={K2[0,0]:.1f}')

# 逐对相对位姿
Rp, Tp = [], []
for i in range(len(used)):
    Rw_l, _ = cv2.Rodrigues(rv1[i]); Rw_r, _ = cv2.Rodrigues(rv2[i])
    Rrel = Rw_r @ Rw_l.T
    Rp.append(Rrel)
    Tp.append((tv2[i] - Rrel @ tv1[i]).ravel())
Rp = np.array(Rp); Tp = np.array(Tp)
bl = np.linalg.norm(Tp, axis=1)
med = np.median(bl)
dev = np.abs(bl - med) / med
# 手持拍摄基线本来就会变(±15% 正常)，只剔除明显崩掉的（>2 倍或 <0.5 倍中位）
inl = (bl <= 2.0 * med) & (bl >= 0.5 * med)
print(f'逐对基线(mm): 中位 {med*1000:.1f}, 范围 {bl.min()*1000:.1f}~{bl.max()*1000:.1f}')
print(f'剔除 {len(used)-inl.sum()} 对异常: {used[~inl].tolist()}')

o2 = [objp] * int(inl.sum())
print('  正在计算全局双目标定（这一步最慢，约 10~30 秒，请稍等）…', flush=True)
rms, K1f, D1f, K2f, D2f, R, T, E, F = cv2.stereoCalibrate(
    o2, [cL_list[i] for i in np.where(inl)[0]], [cR_list[i] for i in np.where(inl)[0]],
    K1, D1, K2, D2, sizeL,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    flags=cv2.CALIB_FIX_INTRINSIC)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1f, D1f, K2f, D2f, sizeL, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
print(f'双目标定 RMS={rms:.3f}px  基线={np.linalg.norm(T)*1000:.1f}mm')

ep_axis = 0 if abs(P2[1, 3]) > abs(P2[0, 3]) else 1
ep = []
for i in range(len(used)):
    u1 = cv2.undistortPoints(cL_list[i], K1f, D1f, R=R1, P=P1).reshape(-1, 2)
    u2 = cv2.undistortPoints(cR_list[i], K2f, D2f, R=R2, P=P2).reshape(-1, 2)
    ep.append(float(np.abs(u1[:, ep_axis] - u2[:, ep_axis]).mean()))
ep = np.array(ep)
print(f'极线残余误差(全局R/T): 中位 {np.median(ep):.2f}px, 最大 {ep.max():.2f}px')

report = dict(left_dir=LEFT_DIR, right_dir=RIGHT_DIR, left_size=sizeL, right_size=sizeR,
              pattern=PATTERN, square_m=SQUARE,
              pairs_all=used.tolist(), pairs_inliers=used[inl].tolist(),
              rms_mono_left=float(rmsL), rms_mono_right=float(rmsR), rms_stereo=float(rms),
              fx_left=float(K1f[0, 0]), fy_left=float(K1f[1, 1]),
              fx_right=float(K2f[0, 0]), fy_right=float(K2f[1, 1]),
              baseline_mm=float(np.linalg.norm(T) * 1000),
              baseline_per_pair_mm=[float(v) * 1000 for v in bl],
              epipolar_error_axis='x' if ep_axis == 0 else 'y',
              epipolar_err_px=[float(v) for v in ep])
with open(os.path.join(OUT, 'calib_report.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

np.savez(os.path.join(OUT, 'stereo_calib_result_pairs.npz'),
         mtx_l=K1f, dist_l=D1f, mtx_r=K2f, dist_r=D2f,
         R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2,
         img_size=np.array(sizeL), right_img_size=np.array(sizeR),
         R_pairs=Rp, T_pairs=Tp, used_pairs=used, inlier_mask=inl, square_m=np.array(SQUARE))
print('\n已保存: results\\stereo_calib_result_pairs.npz  results\\calib_report.json')
