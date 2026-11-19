r"""一键跑完整条流程（在 Visual Studio 里按 F5 就是跑这个）。

和之前的版本区别：三个步骤都在**同一个进程里**顺序执行（用 runpy），
所以 print 的输出会直接出现在 VS 的「输出」窗口里；
以前用 subprocess 时子进程的输出不经过 VS，看起来像“没反应”。

可调参数在最下面。想只跑其中一步，把 RUN_* 改为 False。
"""
import os
import runpy
import sys
import time
import traceback

try:                                     # 防止 VS 输出窗口用非 UTF-8 编码时报 UnicodeEncodeError
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))

RUN_CALIB = True
RUN_BATCH = True
RUN_FUSE = True

RECT_W = 960      # 校正宽度（1440 更密，慢 3~4 倍）
TOPN = 6          # 批量里输出点云的图对数量
FUSE_ALL = True   # True=融合全部图对；False=只融合前 TOPN 对
VOXEL_MM = 4      # 融合后降采样体素(mm)
SHOW_RESULT = True  # 跑完自动弹窗显示结果图（按任意键关闭）


def imread_u(p):
    """cv2.imread 读不了中文路径，统一走 imdecode。"""
    import numpy as np
    import cv2
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def show_images(paths, title='结果预览（按任意键关闭）', width=1300):
    """把结果图拼成一张竖版图弹窗显示；同时另存 results\\preview_montage.png。"""
    import cv2
    ims = []
    for p in paths:
        if not os.path.exists(p):
            continue
        im = imread_u(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        if w != width:
            im = cv2.resize(im, (width, int(round(h * width / w))))
        ims.append(im)
    if not ims:
        print('（没有可显示的结果图）')
        return
    gap = 8
    W = max(i.shape[1] for i in ims)
    H = sum(i.shape[0] for i in ims) + gap * (len(ims) - 1)
    import numpy as np
    canvas = np.full((H, W, 3), 18, np.uint8)
    y = 0
    for im in ims:
        canvas[y:y + im.shape[0], :im.shape[1]] = im
        y += im.shape[0] + gap
    out = os.path.join(HERE, 'results', 'preview_montage.png')
    cv2.imwrite(out, canvas)
    print('结果预览图已保存: ' + out)
    if not SHOW_RESULT:
        return
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, canvas)
        print('已弹窗显示结果，按任意键关闭窗口…', flush=True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print('弹窗失败(%s)，改用系统图片查看器打开' % e)
        try:
            os.startfile(out)
        except Exception as e2:
            print('也无法调用系统查看器: %s' % e2)



def run_step(title, script, args=()):
    print('')
    print('=' * 72)
    print('== ' + title)
    print('=' * 72, flush=True)
    path = os.path.join(HERE, script)
    old_argv = sys.argv
    sys.argv = [path] + [str(a) for a in args]
    t0 = time.time()
    code = 0
    try:
        runpy.run_path(path, run_name='__main__')
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 0
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        sys.argv = old_argv
    print('[%s] %s 用时 %.0fs，退出码 %d' % ('OK' if code == 0 else 'FAIL', title, time.time() - t0, code),
          flush=True)
    return code


def list_pairs():
    d = os.path.join(HERE, 'pairs', 'left')
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        if f.lower().endswith('.jpg'):
            digits = ''.join(c for c in f if c.isdigit())
            if digits:
                out.append(int(digits))
    return sorted(out)


def main():
    print('工作目录: ' + HERE)
    print('Python  : ' + sys.executable)

    if RUN_CALIB:
        if run_step('标定 stereo_calib_pairs.py', 'stereo_calib_pairs.py') != 0:
            return 1
    else:
        print('跳过标定')

    if RUN_BATCH:
        if run_step('批量 batch_pairs.py', 'batch_pairs.py', [RECT_W, TOPN]) != 0:
            return 1
    else:
        print('跳过批量')

    if RUN_FUSE:
        pairs = list_pairs() if FUSE_ALL else []
        args = [RECT_W, VOXEL_MM] + pairs
        if run_step('融合 fuse_clouds.py（%d 对）' % (len(pairs) or TOPN), 'fuse_clouds.py', args) != 0:
            return 1
    else:
        print('跳过融合')

    print('')
    print('全部完成。结果都在 results 目录：')
    print('  results\\stereo_calib_result_pairs.npz   标定参数')
    print('  results\\calib_report.json               标定指标')
    print('  results\\batch\\report.md                 逐对质量表')
    print('  results\\batch\\best*.ply                 前 %d 对点云' % TOPN)
    print('  results\\fused\\fused_pointcloud.ply       融合点云')

    # 跑完自动显示结果图（原来的脚本里没有 imshow，所以 F5 不会弹窗）
    import glob as _glob
    imgs = [os.path.join(HERE, 'results', 'fused', 'fused_view1.png'),
            os.path.join(HERE, 'results', 'fused', 'fused_view2.png')]
    imgs += sorted(_glob.glob(os.path.join(HERE, 'results', 'batch', 'contact_sheet.png')))
    show_images(imgs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
