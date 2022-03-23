import numpy as np
class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()
        # content 就是一个字符串，根据空格分隔开来
        lines = content.split()

        # 去掉空字符
        lines = list(filter(lambda x: len(x), lines))

        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(lines[3])

        self.bbox = [lines[4], lines[5], lines[6], lines[7]]
        self.bbox = np.array([float(x) for x in self.bbox])
        self.dimensions = [lines[8], lines[9], lines[10]]
        self.dimensions = np.array([float(x) for x in self.dimensions])
        self.location = [lines[11], lines[12], lines[13]]
        self.location = np.array([float(x) for x in self.location])
        self.rotation_y = float(lines[14])
        #这一行是模型训练后的label通常最后一行是阈值，可以同个这个过滤掉概率低的object
        #如果只要显示kitti本身则不需要这一行
        #self.ioc = float(lines[15])

