# kitti_vis使用
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
## step4 运行可视化
 可视化点云
```
python one_bin_show.py --index 10
```
![image](https://github.com/zydczqcnydx/kitti_vis/blob/main/vis_result_imgs/BEV.png)
 可视化图片

```
python img_3dbox.py --path_dataset data/object
```
#后面为数据集的位置,/为linux文件格式，windows 为\\\即可

运行图片可视化成功后，该文件为对数据集内的图片进行可视化，默认从数据集第一张开始，可以通过a和d来切换图片。

# 工程结构介绍
1.data文件夹放入的是包括数据集解析的3个类，分别为calib.py,kitti_Dataset.py,object3d.py。他们能够从txt文件中读入我们需要的数据信息。

2.object文件夹内有两个文件夹，其中kitti为官网下载的数据集的部分。training为我们模型训练的结果。

3.可以通过在one_bin_show.py和img_3dbox.py中修改路径来可视化这两个文件夹内的内容.

4.需要注意的是，我们训练的结果中的label中的.txt文件比kitii数据集多了一列置信度。当你要可视化我们的结果时，需要在one_bin_show.py 86行附近进行修改，即可。

## 同时显示点云和图片
```
python final.py --path_dataset data/object
```
1.通过运行上述代码，可以通过opencv和open3d对图片和点云同时显示。

2.运行后会弹出opencv和open3d窗口，可通过a,d进行切换，通过q退出，但是要点击opencv窗口才可以，因为切换图片和点云的索引采用的是cv2.waitKey().

**注意：**第一个点云文件被阻塞住了，需要点击右上角的关闭窗口按钮,程序才能在while中执行，这样就可以在opencv框中，通过a,d进行切换，q退出。
