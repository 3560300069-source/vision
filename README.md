# 双目标定 + 三维重建（左右分开拍照版）

这个目录里有两套脚本：

| 脚本 | 适用情况 |
|---|---|
| `calibrate_new.py` / `constructure3d_new.py` … | 原来的版本：一台相机输出**左右拼接**的一张图（3840×1080），脚本自己切两半 |
| `stereo_calib_pairs.py` / `run_pair.py` | 新加：**左右两台相机分开拍**，各自一个文件夹 |

## 在 Visual Studio 里跑

`run_all.py` 是启动文件：**按 F5 一键跑完 标定 → 批量 → 融合**，约 2~3 分钟。
输出在「视图 → 输出」窗口（下拉选“调试”或“常规”）。三个步骤都在同一进程里顺序执行，
所以 print 能看到；**不要连按两次 F5**，那会起两个实例互相拖慢。

可调开关在 `run_all.py` 顶部：`RUN_CALIB / RUN_BATCH / RUN_FUSE / RECT_W / TOPN / FUSE_ALL / VOXEL_MM`。
只想跑单步，就右键对应脚本 → 调试 → 启动新实例（参数在 项目属性 → 调试 → 脚本参数）。

## 数据摆放

```
vision3d\
  pairs\left\*.jpg     左相机照片（文件名数字要和右边对应）
  pairs\right\*.jpg    右相机照片
  results\             脚本自动生成
```

也可以直接指定目录：`python stereo_calib_pairs.py D:\a D:\b`

## 在 Visual Studio 里怎么跑

1. 打开 `vision3d.sln`（解决方案资源管理器里能看到全部 .py）。
2. 右键 `stereo_calib_pairs.py` → “使用调试启动 / 设为启动文件”，F5。
   控制台会打印每对图的角点检测结果、单目 RMS、基线、极线误差，最后生成
   `results\stereo_calib_result_pairs.npz` 和 `results\calib_report.json`。
3. 再右键 `run_pair.py` → F5（默认第 1 对）。想看别的图对，在
   项目属性 → 调试 → 脚本参数 里填 `11` 或 `11 1440`（第 11 对，校正宽度 1440）。
4. 输出：`results\` 下的校正图、视差伪彩图、`pointcloud_pairXX.ply`（单位米，
   可直接用 MeshLab / CloudCompare / open3d 打开）和两张渲染预览图。

依赖只有 `opencv-python` 和 `numpy`（open3d 只在原来的 constructure*.py 里需要）。

## 换成自己的照片

1. 把自己的照片放进 `pairs\left\`、`pairs\right\`，**文件名用数字对应**（`1.jpg`、`2.jpg`…），左右数量一致。
   旧的示例照片直接删掉或移走。
2. 重新跑 `stereo_calib_pairs.py`。角点缓存带照片签名，换图后会自动失效并重新检测，不用手动删
   `results\corners.npz`。
3. 再跑 `batch_pairs.py` 看逐对质量，或 `run_pair.py <对号>` 看单对点云。

拍照要求：棋盘格完整出现在左右两张画面里、别抖动、两台相机尽量同一时刻触发。

## 参数

`stereo_calib_pairs.py` 顶部：

```python
PATTERN = (9, 6)    # 棋盘格内角点数
SQUARE  = 0.025     # 方格真实边长(米)，实测 25 mm
```

`SQUARE` 直接决定三维尺度：写成 0.025 而实际是 0.02，所有深度/点云坐标会
按 25/20 的比例偏大。改完要重新跑标定（角点缓存与尺度无关，删不删都能复用）。

## 已知限制

- 左右分辨率可以不同（脚本会各自标定，并把低分辨率那台的焦距作为校正基准）。
- 两台相机**必须刚性固定并且同时触发**。若手持、分开按快门，棋盘格在两次曝光
  之间会动，标定出的相对位姿只对棋盘成立，对场景不成立，稠密视差会大量失败。
- 右相机照片若经过微信压缩（960×1280 这种），细节丢失严重，视差会明显变稀；
  尽量以“文件/压缩包”方式传原图。
