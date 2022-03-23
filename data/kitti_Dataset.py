import os
import numpy as np
from data.calib import Calib
from data.object3d import Object3d
import cv2

class Kitti_Dataset:
    def __init__(self, dir_path, split="training"):
        super(Kitti_Dataset, self).__init__()
        self.dir_path = os.path.join(dir_path, split)
        # calib矫正参数文件夹地址
        self.calib = os.path.join(self.dir_path, "calib")
        # RGB图像的文件夹地址
        self.images = os.path.join(self.dir_path, "img")
        # 点云图像文件夹地址
        self.pcs = os.path.join(self.dir_path, "velodyne")
        # 标签文件夹的地址
        self.labels = os.path.join(self.dir_path, "label")

    # 得到当前数据集的大小
    def __len__(self):
        file = []
        for _, _, file in os.walk(self.images):
            pass

        # 返回rgb图片的数量
        return len(file)

    # 得到矫正参数的信息
    def get_calib(self, index):
        # 得到矫正参数文件
        calib_path = os.path.join(self.calib, "{:06d}.txt".format(index))
        with open(calib_path) as f:
            lines = f.readlines()

        lines = list(filter(lambda x: len(x) and x != '\n', lines))
        dict_calib = {}
        for line in lines:
            key, value = line.split(":")
            dict_calib[key] = np.array([float(x) for x in value.split()])
        return Calib(dict_calib)

    def get_rgb(self, index):
        # 首先得到图片的地址
        img_path = os.path.join(self.images, "{:06d}.png".format(index))
        return cv2.imread(img_path)

    def get_pcs(self, index):
        pcs_path = os.path.join(self.pcs, "{:06d}.bin".format(index))
        # 点云的四个数据（x, y, z, r)
        aaa = np.fromfile(pcs_path, dtype=np.float32, count=-1).reshape([-1, 4])
        return aaa[:, :3]

    def get_labels(self, index):
        labels_path = os.path.join(self.labels, "{:06d}.txt".format(index))
        with open(labels_path) as f:
            lines = f.readlines()
        lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))

        return [Object3d(x) for x in lines]
