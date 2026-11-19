r"""结果图弹窗工具：把若干张结果图拼成一张竖版图显示，按任意键关闭。

被 run_pair.py / batch_pairs.py / fuse_clouds.py 共用。
如果设置了环境变量 DSH_NOSHOW=1，则只保存不弹窗（方便无人值守运行/测试）。
"""
import os

import cv2
import numpy as np


def imread_u(p):
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _montage(paths, width=1200):
    ims = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        im = imread_u(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        if w != width:
            im = cv2.resize(im, (width, int(round(h * width / w))))
        ims.append(im)
    if not ims:
        return None
    gap = 8
    W = max(i.shape[1] for i in ims)
    H = sum(i.shape[0] for i in ims) + gap * (len(ims) - 1)
    canvas = np.full((H, W, 3), 18, np.uint8)
    y = 0
    for im in ims:
        canvas[y:y + im.shape[0], :im.shape[1]] = im
        y += im.shape[0] + gap
    return canvas


def show_images(paths, title='结果（按任意键关闭）', width=1200, save_to=None):
    """paths 里有几张就拼几张；弹窗失败则退回系统看图器。"""
    canvas = _montage(paths, width)
    if canvas is None:
        print('（没有可显示的结果图）')
        return None
    if save_to:
        try:
            cv2.imwrite(save_to, canvas)
            print('结果拼图已保存:', save_to)
        except Exception as e:
            print('保存拼图失败:', e)
    if os.environ.get('DSH_NOSHOW') == '1':
        print('（DSH_NOSHOW=1，跳过弹窗）')
        return canvas
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, canvas)
        print('已弹窗显示结果，按任意键关闭…', flush=True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print('弹窗失败(%s)，改用系统看图器打开' % e)
        target = save_to
        if not target or not os.path.exists(target):
            print('（没有保存路径，无法打开）')
            return canvas
        try:
            os.startfile(target)
        except Exception as e2:
            print('也无法打开系统查看器: %s' % e2)
    return canvas
