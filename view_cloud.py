r"""用 open3d 交互式查看点云（拖动旋转、滚轮缩放、Shift+拖动平移）。

用法：
    python view_cloud.py                                  # 默认看融合点云，1cm 降采样
    python view_cloud.py results\fused\fused_pointcloud.ply
    python view_cloud.py results\batch\best01_pair08.ply 0.005
    python view_cloud.py results\fused\fused_pointcloud.ply 0   # 0 = 不降采样（点多会卡）

注意：点云单位是「米」。窗口里按 H 可看快捷键，按 P 看坐标。
"""
import os
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit('没有 open3d，先装：python -m pip install open3d')

HERE = os.path.dirname(os.path.abspath(__file__))

STATS_ONLY = '--stats' in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith('--')]

path = argv[0] if argv else os.path.join(HERE, 'results', 'fused', 'fused_pointcloud.ply')
if not os.path.isabs(path):
    path = os.path.join(HERE, path)
voxel = float(argv[1]) if len(argv) > 1 else 0.01

if not os.path.exists(path):
    raise SystemExit('找不到点云文件: %s' % path)

print('读取:', path)
pcd = o3d.io.read_point_cloud(path)
n0 = len(pcd.points)
print('点数: %d' % n0)
if n0 == 0:
    raise SystemExit('点云是空的，检查文件')

if voxel > 0 and n0 > 300000:
    pcd = pcd.voxel_down_sample(voxel)
    print('降采样到 %.3f m 后: %d 点' % (voxel, len(pcd.points)))

pts = np.asarray(pcd.points)
print('包围盒(mm): %.0f x %.0f x %.0f' % tuple((pts.max(axis=0) - pts.min(axis=0)) * 1000))
print('中心(mm): (%.0f, %.0f, %.0f)' % tuple(pts.mean(axis=0) * 1000))

if STATS_ONLY:
    print('（--stats 模式：只看统计，不开窗口）')
    sys.exit(0)

# 世界坐标系：让 Z 朝上，符合直觉
center = pts.mean(axis=0)
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
frame.translate(center)

print('开窗中：左键旋转 / 滚轮缩放 / Shift+左键平移 / 按 q 或 Esc 关闭')
o3d.visualization.draw_geometries(
    [pcd, frame],
    window_name='点云查看器（单位：米）  %s' % os.path.basename(path),
    width=1280, height=900,
    point_show_normal=False)
