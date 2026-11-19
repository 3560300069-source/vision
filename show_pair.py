r"""按键/轮播查看某一对的结果图（每种图独立显示，不拼在一起）。

先跑一次 run_pair.py 生成图片：
    python run_pair.py 38

然后：
    python show_pair.py            # 自动轮播当前已有结果的图对（每 1.5 秒一张）
    python show_pair.py 38         # 指定图对，自动轮播
    python show_pair.py 38 3       # 只看第 3 种（视差图）
    python show_pair.py --list      # 列出有哪些图

窗口按键：
    1 校正左图   2 校正右图   3 视差   4 点云视角1   5 点云视角2   6 三合一
    p 用 open3d 打开三维点云     n 下一对   b 上一对

注意：按键前先用鼠标点一下图片窗口（点到控制台里会触发“按任意键继续”，程序会退出）。
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')

DEFAULT_PAIR = 38       # 不指定参数时默认看的图对（文件号）
SLIDESHOW = True       # True: 自动轮播（不用按键）；False: 只等按键
SLIDE_MS = 1500        # 轮播每张停留时间
SHOW_W = 1100          # 显示宽度

KEYS = [
    ('1', 'rect_left_%02d.png', '校正-左'),
    ('2', 'rect_right_%02d.png', '校正-右'),
    ('3', 'disp_%02d.png', '视差伪彩'),
    ('4', 'cloud_view1_%02d.png', '点云-视角1'),
    ('5', 'cloud_view2_%02d.png', '点云-视角2'),
    ('6', 'pair_%02d_overview.png', '三合一对比'),
]


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def views(pair):
    out = []
    for k, pat, name in KEYS:
        p = os.path.join(OUT, pat % pair)
        out.append((k, name, p, os.path.exists(p)))
    return out


def available_pairs():
    out = []
    if os.path.isdir(OUT):
        for f in os.listdir(OUT):
            if f.startswith('rect_left_') and f.endswith('.png'):
                d = ''.join(c for c in f[len('rect_left_'):-4] if c.isdigit())
                if d:
                    out.append(int(d))
    return sorted(out)


def show(pair, key):
    for k, name, p, ok in views(pair):
        if k != key:
            continue
        if not ok:
            print('缺少图片: %s（先跑 run_pair.py %d）' % (p, pair))
            placeholder('第 %d 对缺少 %s\n先跑一次 run_pair.py %d' % (pair, name, pair))
            return False
        img = imread_u(p)
        if img is None:
            print('读不出图片:', p)
            return False
        h, w = img.shape[:2]
        if w > SHOW_W:
            img = cv2.resize(img, (SHOW_W, int(round(h * SHOW_W / w))))
        title = 'pair %02d  %s   (1-6切换 p点云 n/b换对 q退出)' % (pair, name)
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, img)
            cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)   # 置顶，避免被 VS 挡住
            cv2.moveWindow(title, 60, 40)
        except Exception as e:
            print('!! 窗口创建失败: %s' % e)
            return False
        print('显示 [%s] %-10s %s' % (k, name, p), flush=True)
        return True
    print('没有按键 %s，可用: %s' % (key, [k for k, _, _, _ in views(pair)]))
    return False


def placeholder(text):
    lines = text.split('\n')
    img = np.full((140 + 70 * len(lines), 980, 3), 30, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (24, 70 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
    cv2.namedWindow('提示', cv2.WINDOW_NORMAL)
    cv2.imshow('提示', img)
    cv2.waitKey(1)


def open_ply(pair):
    import glob
    cands = glob.glob(os.path.join(OUT, 'pointcloud_pair%02d.ply' % pair)) or \
        glob.glob(os.path.join(OUT, 'pointcloud_pair%d.ply' % pair))
    if not cands:
        print('没有 pointcloud_pair%02d.ply，先跑 run_pair.py %d' % (pair, pair))
        return
    try:
        import open3d as o3d
    except ImportError:
        print('没有 open3d，无法打开点云')
        return
    pcd = o3d.io.read_point_cloud(cands[0])
    print('open3d 打开:', cands[0], '点数', len(pcd.points))
    o3d.visualization.draw_geometries([pcd], width=1200, height=900,
                                      window_name='pair %02d 点云（单位：米）' % pair)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    avail = available_pairs()

    if '--list' in sys.argv:
        pair = int(args[0]) if args else (DEFAULT_PAIR if DEFAULT_PAIR in avail else (avail[0] if avail else 1))
        print('现有结果的图对: %s' % avail)
        print('第 %d 对:' % pair)
        for k, name, p, ok in views(pair):
            print('  [%s] %-12s %s  %s' % (k, name, 'OK ' if ok else '缺 ', p))
        return

    pair = int(args[0]) if args else (DEFAULT_PAIR if DEFAULT_PAIR in avail else (avail[0] if avail else 1))
    only = args[1] if len(args) > 1 else None
    print('=' * 60)
    print('show_pair.py 启动了')
    print('  脚本目录:', HERE)
    print('  结果目录:', OUT, '(存在)' if os.path.isdir(OUT) else '(不存在!)')
    print('  已有结果的图对:', avail)
    print('  本次要看的图对:', pair)
    print('=' * 60, flush=True)

    if os.environ.get('DSH_NOSHOW') == '1':
        print('（DSH_NOSHOW=1，只打印不显示）图对 %d' % pair)
        for k, name, p, ok in views(pair):
            print('  [%s] %s %s' % (k, 'OK ' if ok else '缺 ', p))
        return

    if only:                                  # 只看一张
        show(pair, only)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    ready = [k for k, _, _, ok in views(pair) if ok]
    if not ready:
        if avail:
            print('第 %d 对还没有结果图。现有图对: %s' % (pair, avail))
        placeholder('第 %d 对还没有结果图\n先跑一次 run_pair.py %d' % (pair, pair))
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
        return

    print('图对 %d 有 %d 张图。%s' % (
        pair, len(ready),
        '自动轮播中（每 %.1f 秒），按 q 停止，1-6 手动切换，p 看点云' % (SLIDE_MS / 1000.0)
        if SLIDESHOW else '按 1-6 切换，q 退出'))

    i = 0
    while True:
        key = ready[i % len(ready)]
        cv2.destroyAllWindows()
        show(pair, key)
        k = cv2.waitKey(SLIDE_MS if SLIDESHOW else 0) & 0xFF
        ch = chr(k) if 32 <= k < 127 else ''
        if ch == 'q' or k == 27:
            break
        if ch in [k2 for k2, _, _, _ in views(pair)]:
            cv2.destroyAllWindows()
            show(pair, ch)
            k2 = cv2.waitKey(0) & 0xFF
            if k2 == 27 or chr(k2) == 'q':
                break
            continue
        if ch == 'p':
            open_ply(pair)
            continue
        if ch == 'n':
            pair += 1
            ready = [k3 for k3, _, _, ok in views(pair) if ok]
            i = 0
            continue
        if ch == 'b':
            pair = max(1, pair - 1)
            ready = [k3 for k3, _, _, ok in views(pair) if ok]
            i = 0
            continue
        i += 1
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
