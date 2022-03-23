# kitti_vis
## step1 conda 创建环境

```
conda create -n open3d python=3.6 
```
open3d推荐的python版本为3.6，我记得其他版本可能会有冲突。

## step2 安装open3d opencv等主要的库

1.启动open3d环境
```
conda activate open3d
```
2.安装open3d，opencv-python等库
```
pip install open3d
pip install opencv-python 
```
## step3 下载代码
```
git clone https://github.com/zydczqcnydx/kitti_vis.git
```
## step3 运行可视化
 可视化点云
```
python one_bin_show.py --index 10
```
 可视化图片

```
python img_3dbox.py --path_dataset data/object
```
#后面为数据集的位置,/为linux文件格式，windows 为\\\即可

运行图片可视化成功后，该文件为对数据集内的图片进行可视化，默认从数据集第一张开始，可以通过a和d来切换图片。
