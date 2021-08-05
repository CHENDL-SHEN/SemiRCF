import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt # 可视化绘图
import glob
import re
import json
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import os
import shutil
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.measure as ms


test_list = ['ping_52_0_0.png','ping_52_0_1.png','ping_52_0_2.png','ping_52_0_3.png','ping_52_1_0.png','ping_52_1_1.png','ping_52_1_2.png','ping_52_1_3.png','ping_52_2_0.png','ping_52_2_1.png','ping_52_2_2.png','ping_52_2_3.png']
contrast_dir = 'instan_img_mmd' #instan_img_gan
min_grain_size = 50 # 最小的晶粒

if not os.path.exists(contrast_dir):
    os.makedirs(contrast_dir)


def one_wide_bound(gan,thred = 50):
    class Point(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def getX(self):
            return self.x

        def getY(self):
            return self.y

    def getGrayDiff(img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))

    def selectConnects(p):
        if p != 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
                        Point(-1, 0)]
        else:
            connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return connects

    def regionGrow(img, seeds, thresh, label_num, p=0):
        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        label = label_num
        connects = selectConnects(p)
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)
            seedMark[currentPoint.x, currentPoint.y] = label
            for i in range(4):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = label
                    seedList.append(Point(tmpX, tmpY))
        return seedMark

    def dele(x,y,instance):
        cla = set()
        s1 = 0
        if x-1>=0:
            if instance[x-1,y] != 255:
                cla.add(instance[x-1,y])
                s1 = instance[x-1,y]
        if x+1<instance.shape[0]:
            if instance[x+1,y] != 255:
                cla.add(instance[x+1,y])
                s1 = instance[x+1,y]
        if y-1>=0:
            if instance[x,y-1] != 255:
                cla.add(instance[x,y-1])
                s1 = instance[x,y-1]
        if y+1<instance.shape[1]:
            if instance[x,y+1] != 255:
                cla.add(instance[x,y+1])
                s1 = instance[x,y+1]
        if len(cla) == 1:
            return s1
        else:
            return 0

    def dele_valit(x,y,lab):
        cla = lab[x,y]

        if x-1>=0:
            if lab[x-1,y] != cla and lab[x-1,y] != 0:
                return False
        if x+1<lab.shape[0]:
            if lab[x+1,y] != cla and lab[x+1,y] != 0:
                return False
        if y-1>=0:
            if lab[x,y-1] != cla and lab[x,y-1] != 0:
                return False
        if y+1<lab.shape[1]:
            if lab[x,y+1] != cla and lab[x,y+1] != 0:
                return False
        if x-1>=0 and y-1>=0:
            if lab[x-1,y-1] != cla and lab[x-1,y-1] != 0:
                return False
        if x-1>=0 and y+1<lab.shape[1]:
            if lab[x-1,y+1] != cla and lab[x-1,y+1] != 0:
                return False
        if x+1<lab.shape[0] and y-1>=0:
            if lab[x+1,y-1] != cla and lab[x+1,y-1] != 0:
                return False
        if x+1<lab.shape[0] and y+1<lab.shape[1]:
            if lab[x+1,y+1] != cla and lab[x+1,y+1] != 0:
                return False
        return True

    bound = gan.copy()
    bound[np.where(bound>thred)] = 255
    bound[np.where(bound<=thred)] = 0
    instance_image = bound.copy()

    instance_idx = 1
    for i in range(instance_image.shape[0]):
        for j in range(instance_image.shape[1]):
            if instance_image[i, j] == 0:
                seeds = [Point(i, j)]
                out = regionGrow(bound, seeds, 0.1, instance_idx)
                instance_image[np.where(out == instance_idx)[0], np.where(out == instance_idx)[1]] = instance_idx
                instance_idx = instance_idx + 1
    for i in range(instance_idx):
        if np.where(instance_image == instance_idx)[0] <= min_grain_size:
            instance_image[np.where(instance_image == instance_idx)[0], np.where(instance_image == instance_idx)[1]] = 0
    instance_image_ = instance_image.copy()


    s1,s2 = np.where(instance_image==255)
    lab = np.zeros(instance_image.shape)
    for idx in range(len(s1)):
        lab[s1[idx],s2[idx]] = dele(s1[idx],s2[idx],instance_image)
    s1,s2 = np.where(lab>0)
    vali_num = 0
    for idx in range(len(s1)):
        if dele_valit(s1[idx],s2[idx],lab):
            instance_image[s1[idx],s2[idx]] = lab[s1[idx],s2[idx]]
            vali_num = vali_num + 1
    iter = 0
    while(vali_num>10):
        iter = iter + 1
        s1, s2 = np.where(instance_image == 255)
        lab = np.zeros(instance_image.shape)
        for idx in range(len(s1)):
            lab[s1[idx], s2[idx]] = dele(s1[idx], s2[idx], instance_image)
        s1, s2 = np.where(lab > 0)
        vali_num = 0
        for idx in range(len(s1)):
            if dele_valit(s1[idx], s2[idx], lab):
                instance_image[s1[idx], s2[idx]] = lab[s1[idx], s2[idx]]
                vali_num = vali_num + 1
    instance_image[np.where(instance_image==255)] = 0
    bound[np.where(instance_image > 0)] = 0
    return bound


def make_instance(src_bound, name):
    src_bound = one_wide_bound(src_bound)
    instance_image = src_bound.copy()


    class Point(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def getX(self):
            return self.x

        def getY(self):
            return self.y

    def getGrayDiff(img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))

    def selectConnects(p):
        if p != 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
                        Point(-1, 0)]
        else:
            connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return connects

    def regionGrow(img, seeds, thresh, label_num, p=0):
        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        label = label_num
        connects = selectConnects(p)
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)
            seedMark[currentPoint.x, currentPoint.y] = label
            for i in range(4):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = label
                    seedList.append(Point(tmpX, tmpY))
        return seedMark

    instance_idx = 1
    for i in range(instance_image.shape[0]):
        for j in range(instance_image.shape[1]):
            if instance_image[i, j] == 0:
                seeds = [Point(i, j)]
                out = regionGrow(src_bound, seeds, 0.1, instance_idx)
                instance_image[np.where(out == instance_idx)[0], np.where(out == instance_idx)[1]] = instance_idx
                instance_idx = instance_idx + 1
    for i in range(instance_idx):
        if np.where(instance_image == instance_idx)[0] <= min_grain_size:
            instance_image[np.where(instance_image == instance_idx)[0], np.where(instance_image == instance_idx)[1]] = 0
    instance_image[np.where(instance_image == 255)[0], np.where(instance_image == 255)[1]] = 0
    rg_property = ms.regionprops(instance_image, intensity_image=None, cache=True)
    s1 = []
    s2 = []
    s3 = []
    for rg_ in rg_property:
        s1.append(rg_.eccentricity)
        s2.append(rg_.orientation)
        s3.append(rg_.area)
    s1 = np.array(s1)  # eccentricity 越接近1越不圆 越接近0越圆
    s2 = np.array(s2)  # orientation  0表示垂直的 +pi/2表示向右 -pi/2表示向左
    s3 = np.array(s3)  # area
    mean_ecc = np.average(s1)
    mean_size = np.sqrt(np.average(s3))
    mean_ori = np.average(s2)
    count = s2.shape[0]
    std_size = np.std(np.sqrt(s3))
    print('mean_ecc:'+ str(mean_ecc)+ ' mean_size:' + str(mean_size) + ' mean_ori:' + str(mean_ori) + ' count:' + str(count) + ' std_size:' + str(std_size))
    '''
    plt.figure(2)
    plt.hist(s2, bins=6, density=1)
    plt.show()
    '''
    # instance_image[np.where(instance_image==24)[0],np.where(instance_image==24)[1]] = 0 #记得这些是从1开始的
    plt.figure(1, figsize=(4, 4))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.imshow(instance_image)
    plt.savefig(contrast_dir + '/inst_' + name)
    plt.show()
    return mean_size, count, std_size
    # plt.show()


if __name__ == '__main__':

    sou_dir = 'J:\PycharmProjects\one_shot_202107\semi_usedunsup10_vgg16_arg_mmd1f\epoch_50'
    #'Proposed' #'J:\PycharmProjects\one_shot_202107\semi_usedunsup10_vgg16_arg_gan1f\epoch_50' 'J:\PycharmProjects\one_shot_202107\semi_usedunsup10_vgg16_arg_mmd1f\epoch_50'

    mean_size_ = []
    count_ = []
    std_size_ = []
    names = os.listdir(sou_dir)
    for name in names:
        img = cv2.imread(sou_dir + '/' + name, 0)
        mean_size, count, std_size = make_instance(img, name)
        mean_size_.append(mean_size)
        count_.append(count)
        std_size_.append(std_size)
    mean_size_avr = np.mean(mean_size_)
    count_avr = np.mean(count_)
    std_size_avr = np.mean(std_size_)
    print('mean_size:' + str(mean_size_avr) + ' count:' + str(count_avr) + ' std_size:' + str(std_size_avr))

