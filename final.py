import cv2
import numpy as np
from data.kitti_Dataset import Kitti_Dataset
import open3d as o3d

import time

from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_dataset', type=str, default=None, help='dir for the label data', required=True)
args = parser.parse_args()

# 根据偏航角计算旋转矩阵（逆时针旋转）
def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R

def draw_3dframeworks(vis,points):

    position = points
    points_box = np.transpose(position)

    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7], [0, 5], [1, 4]])
    colors = np.array([[1., 0., 1.] for j in range(len(lines_box))])
    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(points_box)
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)


    vis.add_geometry(line_set)
    render_option = vis.get_render_option()
    render_option.point_size = 3
    render_option.background_color = np.asarray([1, 1, 1])
    # vis.get_render_option().load_from_json('renderoption.json')
    param = o3d.io.read_pinhole_camera_parameters('BV_1440.json')

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.update_renderer()

if __name__ == "__main__":


    dir_path = Path(args.path_dataset)
    # 读取训练集文件夹
    split = "training"
    dataset = Kitti_Dataset(dir_path, split=split)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1440, height=1080)

    index = 0
    max_num = 100

    # 逐张读入图片
    while True:
        img3_d = dataset.get_rgb(index)
        calib = dataset.get_calib(index)
        # 获取标签数据
        obj = dataset.get_labels(index)

        img3_d = dataset.get_rgb(index)
        calib1 = dataset.get_calib(index)
        pc = dataset.get_pcs(index)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc)
        point_cloud.paint_uniform_color([0, 121 / 255, 89 / 255])
        vis.add_geometry(point_cloud)

        # 逐个读入一副图片中的所有object的标签
        for num in range(len(obj)):
            if obj[num].name == "Car" or obj[num].name == "Pedestrian" or obj[num].name == "Cyclist":
                # 阈值设置 ioc
                if (obj[num].name == "Car" and obj[num].ioc >= 0.5) or obj[num].ioc > 0.5:
                    point_cloud = o3d.geometry.PointCloud()
                    # step1 得到rot_y旋转矩阵 3*3
                    R = rot_y(obj[num].rotation_y)
                    # 读取obect物体的高宽长信息
                    h, w, l = obj[num].dimensions[0], obj[num].dimensions[1], obj[num].dimensions[2]

                    x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                    y = [0, 0, 0, 0, -h, -h, -h, -h]
                    z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
                    # 将xyz转化成3*8的矩阵
                    corner_3d = np.vstack([x, y, z])
                    # R * X
                    corner_3d = np.dot(R, corner_3d)

                    # 将该物体移动到相机坐标系下的原点处（涉及到坐标的移动，直接相加就行）
                    corner_3d[0, :] += obj[num].location[0]
                    corner_3d[1, :] += obj[num].location[1]
                    corner_3d[2, :] += obj[num].location[2]

                    # 将3d的bbox转换到2d坐标系中（需要用到内参矩阵)
                    corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
                    corner_2d = np.dot(calib.P2, corner_3d)
                    # 在像素坐标系下，横坐标x = corner_2d[0, :] /= corner_2d[2, :]
                    # 纵坐标的值以此类推
                    corner_2d[0, :] /= corner_2d[2, :]
                    corner_2d[1, :] /= corner_2d[2, :]

                    corner_2d = np.array(corner_2d, dtype=np.int)

                    # 三维坐标
                    corner_3d[-1][-1] = 1
                    inv_Tr = np.zeros_like(calib.Tr_velo_to_cam)
                    inv_Tr[0:3, 0:3] = np.transpose(calib.Tr_velo_to_cam[0:3, 0:3])
                    inv_Tr[0:3, 3] = np.dot(-np.transpose(calib.Tr_velo_to_cam[0:3, 0:3]), calib.Tr_velo_to_cam[0:3, 3])

                    Y = np.dot(inv_Tr, corner_3d)
                    draw_3dframeworks(vis, Y)

                    # 绘制立方体边界框
                    color = [255, 0, 255]
                    # 线宽
                    thickness = 2
                    if corner_2d.min() >= 0:
                        #绘制3d框
                        for corner_i in range(0, 4):
                            i, j = corner_i, (corner_i + 1) % 4
                            cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)
                            i, j = corner_i + 4, (corner_i + 1) % 4 + 4
                            cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)
                            i, j = corner_i, corner_i + 4
                            cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)


                        cv2.line(img3_d,(corner_2d[0, 0],corner_2d[1, 0]), (corner_2d[0, 5], corner_2d[1, 5]),color, thickness)
                        cv2.line(img3_d, (corner_2d[0, 1], corner_2d[1, 1]), (corner_2d[0, 4], corner_2d[1, 4]), color, thickness)
        cv2.imshow("3dbox_img", img3_d)
        vis.run()

        vis.clear_geometries()
        key = cv2.waitKey(100) & 0xFF
        if key == ord('d'):
            index += 1
        if key == ord('a'):
            index -= 1
        if key == ord('q'):
            break
        if index >= max_num:
            index = max_num - 1
        if index < 0:
            index = 0
        # 读入图片信息
        print(index)

