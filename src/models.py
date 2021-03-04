import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import EdgeGenerator
from .loss import cross_entropy_loss_RCF, cross_entropy_loss_perued_RCF
import scipy.io as sio
import numpy as np
import cv2
from scipy import signal
from sklearn.cluster import KMeans
import re
from torch.optim import lr_scheduler

def load_vgg16pretrain(model, vggmodel='vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)
    print('loading VGG')

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        #########################################
        #self.name = 'EdgeModel_2000_' #11500
        #########################################
        self.config = config
        self.iteration = 0

        self.gen_weights_path = config.load_model_dir


    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            print('loding: ' + self.gen_weights_path)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)
            if 'RCF-pytorch-master' in self.gen_weights_path:
                data['state_dict']['score_dsn1s.weight'] = data['state_dict']['score_dsn1.weight']
                data['state_dict']['score_dsn2s.weight'] = data['state_dict']['score_dsn2.weight']
                data['state_dict']['score_dsn3s.weight'] = data['state_dict']['score_dsn3.weight']
                data['state_dict']['score_dsn4s.weight'] = data['state_dict']['score_dsn4.weight']
                data['state_dict']['score_dsn5s.weight'] = data['state_dict']['score_dsn5.weight']
                data['state_dict']['score_dsn1s.bias'] = data['state_dict']['score_dsn1.bias']
                data['state_dict']['score_dsn2s.bias'] = data['state_dict']['score_dsn2.bias']
                data['state_dict']['score_dsn3s.bias'] = data['state_dict']['score_dsn3.bias']
                data['state_dict']['score_dsn4s.bias'] = data['state_dict']['score_dsn4.bias']
                data['state_dict']['score_dsn5s.bias'] = data['state_dict']['score_dsn5.bias']
                data['state_dict']['score_finals.weight'] = data['state_dict']['score_final.weight']
                data['state_dict']['score_finals.bias'] = data['state_dict']['score_final.bias']
                self.generator.load_state_dict(data['state_dict'])
                print('loading: old 12: overrid score layer')


            else:
                self.generator.load_state_dict(data['generator'])





            if 'EdgeModel' in self.gen_weights_path:
                sss =  self.gen_weights_path.split('EdgeModel')[1]
                if len(re.findall("\d+", sss)) == 0:
                    iteration = 0
                else:
                    iteration = int(re.findall("\d+", sss)[0])
            else:
                iteration = 0

            self.iteration = iteration
        else:
            self.generator.apply(weights_init)
            load_vgg16pretrain(self.generator)

    def save(self):
        print('\nsaving %s...\n' % self.name)

        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, os.path.join(self.config.PATH, self.name + '_'+ str(self.iteration) + '_' +'_gen.pth'))


def build_label_process(arg):

    def get_route(weight_matrix, x_y_same=True):
        class point():
            def __init__(self, x, y, cost, route):
                self.x = x
                self.y = y
                self.cost = cost
                self.route = route

        def ini_poinstd(weight_matrix):
            dis_hei = weight_matrix.shape[0]
            dis_wid = weight_matrix.shape[1]
            list = []
            for i in range(dis_hei):
                list_tmp = []
                for j in range(dis_wid):
                    list_tmp.append(point(i, j, weight_matrix[i, j], [(i, j)]))
                list.append(list_tmp)
            return list

        dis_hei = weight_matrix.shape[0]
        dis_wid = weight_matrix.shape[1]
        if x_y_same:
            weight_matrix = weight_matrix
        else:
            weight_matrix = weight_matrix[:, ::-1]

        list = ini_poinstd(weight_matrix)
        route = np.zeros([dis_hei, dis_wid])

        for dis in range(1, dis_hei + dis_wid - 1):
            for dis_i in range(0, dis + 1):
                x = dis_i
                y = dis - dis_i

                if x >= dis_hei or y >= dis_wid:
                    continue

                if x - 1 >= 0:
                    cost_1 = list[x - 1][y].cost
                else:
                    cost_1 = -1e6
                if y - 1 >= 0:
                    cost_2 = list[x][y - 1].cost
                else:
                    cost_2 = -1e6

                if cost_1 > cost_2:
                    list[x][y].cost = list[x][y].cost + cost_1
                    list[x][y].route = list[x][y].route + list[x - 1][y].route
                elif cost_1 < cost_2:
                    list[x][y].cost = list[x][y].cost + cost_2
                    list[x][y].route = list[x][y].route + list[x][y - 1].route

        route_list = list[dis_hei - 1][dis_wid - 1].route
        route_cost = list[dis_hei - 1][dis_wid - 1].cost - weight_matrix[0, 0] - weight_matrix[dis_hei - 1, dis_wid - 1]
        for i in route_list:
            route[i] = 1

        if x_y_same:
            route = route
        else:
            route = route[:, ::-1]

        return route, route_cost

    def min_dis_get(e1_jihe, e2_jihe, center_x, center_y):
        # 返回 最小的序号 最小的距离

        opt_idx = -1
        min_dis_opt = 10e6

        for e1_idx in range(len(e1_jihe)):
            dis_tmp = np.sqrt((e1_jihe[e1_idx] - center_x) ** 2 + (e2_jihe[e1_idx] - center_y) ** 2)
            if min_dis_opt > dis_tmp:
                min_dis_opt = dis_tmp
                opt_idx = e1_idx

        return opt_idx, min_dis_opt

    def min_dis_getjihe(e1_jihe, e2_jihe, center_x, center_y):
        # 返回 距离

        jihe_dis = []

        for e1_idx in range(len(e1_jihe)):
            dis_tmp = np.sqrt((e1_jihe[e1_idx] - center_x) ** 2 + (e2_jihe[e1_idx] - center_y) ** 2)
            jihe_dis.append(dis_tmp)

        return np.array(jihe_dis)

    def getGrayDiff(img, currentPoint, tmpPoint):
        return np.abs(img[currentPoint.x, currentPoint.y] - img[tmpPoint.x, tmpPoint.y])

    class Point(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def getX(self):
            return self.x

        def getY(self):
            return self.y

    def selectConnects(p):
        if p == 2:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                        Point(-1, 1),
                        Point(-1, 0), Point(-2, 0), Point(2, 0), Point(0, -2), Point(0, 2)]
        elif p == 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                        Point(-1, 1),
                        Point(-1, 0)]
        else:
            connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return connects

    def regionGrow(img, seed, label, p=0):
        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        seedList = []
        seedList.append(seed)
        connects = selectConnects(p)
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)
            seedMark[currentPoint.x, currentPoint.y] = label
            num = 8
            for i in range(num):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                if grayDiff == 0 and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = label
                    seedList.append(Point(tmpX, tmpY))
        return seedMark

    def pos_neibour(edges, kernal=3, method='normal'):
        if method == 'normal':
            ones = np.ones([kernal, kernal])
        elif method == 'dis':
            ones = np.ones([kernal, kernal])
            center = int(kernal / 2)
            for x in range(kernal):
                for y in range(kernal):
                    if (x - center) ** 2 + (y - center) ** 2 > center ** 2:
                        ones[x, y] = 0

        edge_numpy = edges
        edge_numpy[np.where(edge_numpy != 1)] = 0
        nbeibour = signal.convolve2d(edge_numpy, ones, boundary='fill', mode='same')
        return nbeibour

    def region_access(edges):
        ones = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1]).reshape([3, 3])
        edge_numpy = edges
        edge_numpy[np.where(edge_numpy != 1)] = 0
        nbeibour = signal.convolve2d(edge_numpy, ones, boundary='fill', mode='same')
        return nbeibour

    def gene_point(positive_advance, edges):
        shuxing = -1 * np.ones(positive_advance.shape)
        nbeibour = np.zeros(positive_advance.shape)
        s1, s2 = np.where(positive_advance == 1)
        for i in range(len(s1)):
            edges_cp = edges.copy()
            fangxiang = np.zeros([4])
            x = s1[i]
            y = s2[i]
            edges_cp[x, y] = 0
            num = 0
            while (num < 30):
                if edges_cp[x - 1, y] == 1:
                    fangxiang[0] = 1
                    x = x - 1
                    num = num + 1
                    edges_cp[x, y] = 0
                elif edges_cp[x, y + 1] == 1:
                    fangxiang[1] = 1
                    y = y + 1
                    num = num + 1
                    edges_cp[x, y] = 0
                elif edges_cp[x + 1, y] == 1:
                    fangxiang[2] = 1
                    x = x + 1
                    num = num + 1
                    edges_cp[x, y] = 0
                elif edges_cp[x, y - 1] == 1:
                    fangxiang[3] = 1
                    y = y - 1
                    num = num + 1
                    edges_cp[x, y] = 0
                else:
                    break
            if fangxiang[0] == 1 and np.sum(fangxiang) == 1:
                shuxing[s1[i], s2[i]] = 1
                nbeibour[s1[i] + 1, s2[i]] = 1
                nbeibour[s1[i], s2[i] - 1] = 1
                nbeibour[s1[i], s2[i] + 1] = 1
                shuxing[s1[i] + 1, s2[i]] = 1
                shuxing[s1[i], s2[i] - 1] = 5
                shuxing[s1[i], s2[i] + 1] = 8
            elif fangxiang[1] == 1 and np.sum(fangxiang) == 1:
                shuxing[s1[i], s2[i]] = 2
                nbeibour[s1[i] - 1, s2[i]] = 1
                nbeibour[s1[i], s2[i] - 1] = 1
                nbeibour[s1[i] + 1, s2[i]] = 1
                shuxing[s1[i] - 1, s2[i]] = 6
                shuxing[s1[i], s2[i] - 1] = 2
                shuxing[s1[i] + 1, s2[i]] = 5
            elif fangxiang[2] == 1 and np.sum(fangxiang) == 1:
                shuxing[s1[i], s2[i]] = 3
                nbeibour[s1[i] - 1, s2[i]] = 1
                nbeibour[s1[i], s2[i] - 1] = 1
                nbeibour[s1[i], s2[i] + 1] = 1
                shuxing[s1[i] - 1, s2[i]] = 3
                shuxing[s1[i], s2[i] - 1] = 6
                shuxing[s1[i], s2[i] + 1] = 7
            elif fangxiang[3] == 1 and np.sum(fangxiang) == 1:
                shuxing[s1[i], s2[i]] = 4
                nbeibour[s1[i] - 1, s2[i]] = 1
                nbeibour[s1[i], s2[i] + 1] = 1
                nbeibour[s1[i] + 1, s2[i]] = 1
                shuxing[s1[i] - 1, s2[i]] = 7
                shuxing[s1[i], s2[i] + 1] = 4
                shuxing[s1[i] + 1, s2[i]] = 8
            elif fangxiang[0] == 1 and fangxiang[1] == 1 and np.sum(fangxiang) == 2:
                shuxing[s1[i], s2[i]] = 5
                nbeibour[s1[i], s2[i] - 1] = 1
                nbeibour[s1[i] + 1, s2[i]] = 1
                shuxing[s1[i], s2[i] - 1] = 5
                shuxing[s1[i] + 1, s2[i]] = 5
            elif fangxiang[1] == 1 and fangxiang[2] == 1 and np.sum(fangxiang) == 2:
                shuxing[s1[i], s2[i]] = 6
                nbeibour[s1[i], s2[i] - 1] = 1
                nbeibour[s1[i] - 1, s2[i]] = 1
                shuxing[s1[i], s2[i] - 1] = 6
                shuxing[s1[i] - 1, s2[i]] = 6
            elif fangxiang[2] == 1 and fangxiang[3] == 1 and np.sum(fangxiang) == 2:
                shuxing[s1[i], s2[i]] = 7
                nbeibour[s1[i], s2[i] + 1] = 1
                nbeibour[s1[i] - 1, s2[i]] = 1
                shuxing[s1[i], s2[i] + 1] = 7
                shuxing[s1[i] - 1, s2[i]] = 7
            elif fangxiang[3] == 1 and fangxiang[0] == 1 and np.sum(fangxiang) == 2:
                shuxing[s1[i], s2[i]] = 8
                nbeibour[s1[i], s2[i] + 1] = 1
                nbeibour[s1[i] + 1, s2[i]] = 1
                shuxing[s1[i], s2[i] + 1] = 8
                shuxing[s1[i] + 1, s2[i]] = 8
            elif np.sum(fangxiang) == 0:
                nbeibour[s1[i] + 1, s2[i]] = 1
                nbeibour[s1[i] - 1, s2[i]] = 1
                nbeibour[s1[i], s2[i] - 1] = 1
                nbeibour[s1[i], s2[i] + 1] = 1
                shuxing[s1[i] + 1, s2[i]] = 1
                shuxing[s1[i] - 1, s2[i]] = 3
                shuxing[s1[i], s2[i] - 1] = 2
                shuxing[s1[i], s2[i] + 1] = 4
        return nbeibour, shuxing

    def pos_neibour_add(nbeibour, x_a, y_b):
        nbeibour[x_a - 1, y_b - 1] = nbeibour[x_a - 1, y_b - 1] + 1
        nbeibour[x_a - 1, y_b] = nbeibour[x_a - 1, y_b] + 1
        nbeibour[x_a - 1, y_b + 1] = nbeibour[x_a - 1, y_b + 1] + 1
        nbeibour[x_a, y_b - 1] = nbeibour[x_a, y_b - 1] + 1
        nbeibour[x_a, y_b] = nbeibour[x_a, y_b] + 1
        nbeibour[x_a, y_b + 1] = nbeibour[x_a, y_b + 1] + 1
        nbeibour[x_a + 1, y_b - 1] = nbeibour[x_a + 1, y_b - 1] + 1
        nbeibour[x_a + 1, y_b] = nbeibour[x_a + 1, y_b] + 1
        nbeibour[x_a + 1, y_b + 1] = nbeibour[x_a + 1, y_b + 1] + 1
        return nbeibour

    def pos_adv_plus(positive_advance, positive, edge, x_a, y_b, shuxing_advance, threhold_pos_adv):
        if shuxing_advance[x_a, y_b] == 1:
            shuxing_advance[x_a + 1, y_b] = 1
            shuxing_advance[x_a, y_b - 1] = 5
            shuxing_advance[x_a, y_b + 1] = 8
            shuxing_advance[x_a - 1, y_b] = -1
        elif shuxing_advance[x_a, y_b] == 2:
            shuxing_advance[x_a - 1, y_b] = 6
            shuxing_advance[x_a, y_b - 1] = 2
            shuxing_advance[x_a + 1, y_b] = 5
            shuxing_advance[x_a, y_b + 1] = -1
        elif shuxing_advance[x_a, y_b] == 3:
            shuxing_advance[x_a - 1, y_b] = 3
            shuxing_advance[x_a, y_b - 1] = 6
            shuxing_advance[x_a, y_b + 1] = 7
            shuxing_advance[x_a + 1, y_b] = -1
        elif shuxing_advance[x_a, y_b] == 4:
            shuxing_advance[x_a - 1, y_b] = 7
            shuxing_advance[x_a, y_b + 1] = 4
            shuxing_advance[x_a + 1, y_b] = 8
            shuxing_advance[x_a, y_b - 1] = -1
        elif shuxing_advance[x_a, y_b] == 5:
            shuxing_advance[x_a, y_b - 1] = 5
            shuxing_advance[x_a + 1, y_b] = 5
            shuxing_advance[x_a - 1, y_b] = -1
            shuxing_advance[x_a, y_b + 1] = -1
        elif shuxing_advance[x_a, y_b] == 6:
            shuxing_advance[x_a, y_b - 1] = 6
            shuxing_advance[x_a - 1, y_b] = 6
            shuxing_advance[x_a + 1, y_b] = -1
            shuxing_advance[x_a, y_b + 1] = -1
        elif shuxing_advance[x_a, y_b] == 7:
            shuxing_advance[x_a, y_b + 1] = 7
            shuxing_advance[x_a - 1, y_b] = 7
            shuxing_advance[x_a + 1, y_b] = -1
            shuxing_advance[x_a, y_b - 1] = -1
        elif shuxing_advance[x_a, y_b] == 8:
            shuxing_advance[x_a, y_b + 1] = 8
            shuxing_advance[x_a + 1, y_b] = 8
            shuxing_advance[x_a - 1, y_b] = -1
            shuxing_advance[x_a, y_b - 1] = -1
        elif shuxing_advance[x_a, y_b] == 0:
            shuxing_advance[x_a + 1, y_b] = 1
            shuxing_advance[x_a - 1, y_b] = 3
            shuxing_advance[x_a, y_b - 1] = 2
            shuxing_advance[x_a, y_b + 1] = 4

        if edge[x_a - 1, y_b] != 1 and shuxing_advance[x_a - 1, y_b] != -1:
            a = 0
        else:
            positive_advance[x_a - 2, y_b] = threhold_pos_adv
            positive_advance[x_a, y_b] = threhold_pos_adv
            positive_advance[x_a - 1, y_b - 1] = threhold_pos_adv
            positive_advance[x_a - 1, y_b + 1] = threhold_pos_adv
        if edge[x_a, y_b - 1] != 1 and shuxing_advance[x_a, y_b - 1] != -1:
            a = 0
        else:
            positive_advance[x_a - 1, y_b - 1] = threhold_pos_adv
            positive_advance[x_a + 1, y_b - 1] = threhold_pos_adv
            positive_advance[x_a, y_b - 2] = threhold_pos_adv
            positive_advance[x_a, y_b] = threhold_pos_adv
        if edge[x_a, y_b + 1] != 1 and shuxing_advance[x_a, y_b + 1] != -1:
            a = 0
        else:
            positive_advance[x_a - 1, y_b + 1] = threhold_pos_adv
            positive_advance[x_a + 1, y_b + 1] = threhold_pos_adv
            positive_advance[x_a, y_b] = threhold_pos_adv
            positive_advance[x_a, y_b + 2] = threhold_pos_adv
        if edge[x_a + 1, y_b] != 1 and shuxing_advance[x_a + 1, y_b] != -1:
            a = 0
        else:
            positive_advance[x_a, y_b] = threhold_pos_adv
            positive_advance[x_a + 2, y_b] = threhold_pos_adv
            positive_advance[x_a + 1, y_b - 1] = threhold_pos_adv
            positive_advance[x_a + 1, y_b + 1] = threhold_pos_adv

        return positive_advance, shuxing_advance

    def get_advance_mask(edges, positive, mask):
        nbeibour = pos_neibour(edges.copy())
        region = region_access(edges.copy())
        n1, n2, n3, n4, n5, n6, n7, n8 = filt_0(edges.copy())
        positive_advance = 0 * np.ones(positive.shape)
        # 选择可生长的点
        positive_advance[np.where(region == 1)] = 1
        positive_advance[np.where(n1 == 2)] = 1
        positive_advance[np.where(n2 == 2)] = 1
        positive_advance[np.where(n3 == 2)] = 1
        positive_advance[np.where(n4 == 2)] = 1
        positive_advance[np.where(n5 == 2)] = 1
        positive_advance[np.where(n6 == 2)] = 1
        positive_advance[np.where(n7 == 2)] = 1
        positive_advance[np.where(n8 == 2)] = 1
        positive_advance[np.where(edges != 1)] = 0
        positive_advance[np.where(mask == 255)] = 0
        positive_advance, shuxing_advance = gene_point(positive_advance, edges.copy())
        positive_advance[np.where(positive_advance > 0)] = positive[np.where(positive_advance > 0)]
        positive_advance[np.where(edges == 1)] = threhold
        return nbeibour, positive_advance, shuxing_advance

    def valit(nbeibour, x_a, y_b):
        if nbeibour[x_a - 1, y_b - 1] >= pos_neibour_max:
            return False
        if nbeibour[x_a - 1, y_b] >= pos_neibour_max:
            return False
        if nbeibour[x_a - 1, y_b + 1] >= pos_neibour_max:
            return False
        if nbeibour[x_a, y_b - 1] >= pos_neibour_max:
            return False
        if nbeibour[x_a, y_b] >= pos_neibour_max:
            return False
        if nbeibour[x_a, y_b + 1] >= pos_neibour_max:
            return False
        if nbeibour[x_a + 1, y_b - 1] >= pos_neibour_max:
            return False
        if nbeibour[x_a + 1, y_b] >= pos_neibour_max:
            return False
        if nbeibour[x_a + 1, y_b + 1] >= pos_neibour_max:
            return False
        return True

    def pos_adv_add(positive_advance, positive, edge, x_a, y_b, shuxing_advance, threhold_pos_adv):
        if shuxing_advance[x_a, y_b] == 1:
            shuxing_advance[x_a + 1, y_b] = 1
            shuxing_advance[x_a, y_b - 1] = 5
            shuxing_advance[x_a, y_b + 1] = 8
            shuxing_advance[x_a - 1, y_b] = -1
        elif shuxing_advance[x_a, y_b] == 2:
            shuxing_advance[x_a - 1, y_b] = 6
            shuxing_advance[x_a, y_b - 1] = 2
            shuxing_advance[x_a + 1, y_b] = 5
            shuxing_advance[x_a, y_b + 1] = -1
        elif shuxing_advance[x_a, y_b] == 3:
            shuxing_advance[x_a - 1, y_b] = 3
            shuxing_advance[x_a, y_b - 1] = 6
            shuxing_advance[x_a, y_b + 1] = 7
            shuxing_advance[x_a + 1, y_b] = -1
        elif shuxing_advance[x_a, y_b] == 4:
            shuxing_advance[x_a - 1, y_b] = 7
            shuxing_advance[x_a, y_b + 1] = 4
            shuxing_advance[x_a + 1, y_b] = 8
            shuxing_advance[x_a, y_b - 1] = -1
        elif shuxing_advance[x_a, y_b] == 5:
            shuxing_advance[x_a, y_b - 1] = 5
            shuxing_advance[x_a + 1, y_b] = 5
            shuxing_advance[x_a - 1, y_b] = -1
            shuxing_advance[x_a, y_b + 1] = -1
        elif shuxing_advance[x_a, y_b] == 6:
            shuxing_advance[x_a, y_b - 1] = 6
            shuxing_advance[x_a - 1, y_b] = 6
            shuxing_advance[x_a + 1, y_b] = -1
            shuxing_advance[x_a, y_b + 1] = -1
        elif shuxing_advance[x_a, y_b] == 7:
            shuxing_advance[x_a, y_b + 1] = 7
            shuxing_advance[x_a - 1, y_b] = 7
            shuxing_advance[x_a + 1, y_b] = -1
            shuxing_advance[x_a, y_b - 1] = -1
        elif shuxing_advance[x_a, y_b] == 8:
            shuxing_advance[x_a, y_b + 1] = 8
            shuxing_advance[x_a + 1, y_b] = 8
            shuxing_advance[x_a - 1, y_b] = -1
            shuxing_advance[x_a, y_b - 1] = -1
        elif shuxing_advance[x_a, y_b] == 0:
            shuxing_advance[x_a + 1, y_b] = 1
            shuxing_advance[x_a - 1, y_b] = 3
            shuxing_advance[x_a, y_b - 1] = 2
            shuxing_advance[x_a, y_b + 1] = 4

        if edge[x_a - 1, y_b] != 1 and shuxing_advance[x_a - 1, y_b] != -1:
            positive_advance[x_a - 1, y_b] = positive[x_a - 1, y_b]
        else:
            positive_advance[x_a - 2, y_b] = threhold_pos_adv
            positive_advance[x_a, y_b] = threhold_pos_adv
            positive_advance[x_a - 1, y_b - 1] = threhold_pos_adv
            positive_advance[x_a - 1, y_b + 1] = threhold_pos_adv
        if edge[x_a, y_b - 1] != 1 and shuxing_advance[x_a, y_b - 1] != -1:
            positive_advance[x_a, y_b - 1] = positive[x_a, y_b - 1]
        else:
            positive_advance[x_a - 1, y_b - 1] = threhold_pos_adv
            positive_advance[x_a + 1, y_b - 1] = threhold_pos_adv
            positive_advance[x_a, y_b - 2] = threhold_pos_adv
            positive_advance[x_a, y_b] = threhold_pos_adv
        if edge[x_a, y_b + 1] != 1 and shuxing_advance[x_a, y_b + 1] != -1:
            positive_advance[x_a, y_b + 1] = positive[x_a, y_b + 1]
        else:
            positive_advance[x_a - 1, y_b + 1] = threhold_pos_adv
            positive_advance[x_a + 1, y_b + 1] = threhold_pos_adv
            positive_advance[x_a, y_b] = threhold_pos_adv
            positive_advance[x_a, y_b + 2] = threhold_pos_adv
        if edge[x_a + 1, y_b] != 1 and shuxing_advance[x_a + 1, y_b] != -1:
            positive_advance[x_a + 1, y_b] = positive[x_a + 1, y_b]
        else:
            positive_advance[x_a, y_b] = threhold_pos_adv
            positive_advance[x_a + 2, y_b] = threhold_pos_adv
            positive_advance[x_a + 1, y_b - 1] = threhold_pos_adv
            positive_advance[x_a + 1, y_b + 1] = threhold_pos_adv

        return positive_advance, shuxing_advance

    def filt_0(edges):
        ones_1 = np.array([3, 1, 1, 3, 0, 3, 3, 3, 3]).reshape([3, 3])
        ones_2 = np.array([1, 1, 3, 3, 0, 3, 3, 3, 3]).reshape([3, 3])
        ones_3 = np.array([3, 3, 3, 3, 0, 3, 1, 1, 3]).reshape([3, 3])
        ones_4 = np.array([3, 3, 3, 3, 0, 3, 3, 1, 1]).reshape([3, 3])
        ones_5 = np.array([1, 3, 3, 1, 0, 3, 3, 3, 3]).reshape([3, 3])
        ones_6 = np.array([0, 3, 3, 1, 0, 3, 1, 3, 3]).reshape([3, 3])
        ones_7 = np.array([3, 3, 1, 3, 0, 1, 3, 3, 3]).reshape([3, 3])
        ones_8 = np.array([3, 3, 3, 3, 0, 1, 0, 0, 1]).reshape([3, 3])
        edge_numpy = edges
        edge_numpy[np.where(edge_numpy != 1)] = 0
        nbeibour_1 = signal.convolve2d(edge_numpy, ones_1, boundary='fill', mode='same')
        nbeibour_2 = signal.convolve2d(edge_numpy, ones_2, boundary='fill', mode='same')
        nbeibour_3 = signal.convolve2d(edge_numpy, ones_3, boundary='fill', mode='same')
        nbeibour_4 = signal.convolve2d(edge_numpy, ones_4, boundary='fill', mode='same')
        nbeibour_5 = signal.convolve2d(edge_numpy, ones_5, boundary='fill', mode='same')
        nbeibour_6 = signal.convolve2d(edge_numpy, ones_6, boundary='fill', mode='same')
        nbeibour_7 = signal.convolve2d(edge_numpy, ones_7, boundary='fill', mode='same')
        nbeibour_8 = signal.convolve2d(edge_numpy, ones_8, boundary='fill', mode='same')
        return nbeibour_1, nbeibour_2, nbeibour_3, nbeibour_4, nbeibour_5, nbeibour_6, nbeibour_7, nbeibour_8

    def get_advance(edges, positive):
        nbeibour = pos_neibour(edges.copy())
        region = region_access(edges.copy())
        n1, n2, n3, n4, n5, n6, n7, n8 = filt_0(edges.copy())
        positive_advance = 0 * np.ones(positive.shape)
        # 选择可生长的点
        positive_advance[np.where(region == 1)] = 1
        positive_advance[np.where(n1 == 2)] = 1
        positive_advance[np.where(n2 == 2)] = 1
        positive_advance[np.where(n3 == 2)] = 1
        positive_advance[np.where(n4 == 2)] = 1
        positive_advance[np.where(n5 == 2)] = 1
        positive_advance[np.where(n6 == 2)] = 1
        positive_advance[np.where(n7 == 2)] = 1
        positive_advance[np.where(n8 == 2)] = 1
        positive_advance[np.where(edges != 1)] = 0
        positive_advance, shuxing_advance = gene_point(positive_advance, edges.copy())
        positive_advance[np.where(positive_advance > 0)] = positive[np.where(positive_advance > 0)]
        positive_advance[np.where(edges == 1)] = threhold
        return nbeibour, positive_advance, shuxing_advance

    def iso_point(edges):
        # -1, 0 , 1 边, 非边, 真端点
        def filt_0(edges):
            ones_1 = np.array([3, 1, 1, 3, 0, 3, 3, 3, 3]).reshape([3, 3])
            ones_2 = np.array([1, 1, 3, 3, 0, 3, 3, 3, 3]).reshape([3, 3])
            ones_3 = np.array([3, 3, 3, 3, 0, 3, 1, 1, 3]).reshape([3, 3])
            ones_4 = np.array([3, 3, 3, 3, 0, 3, 3, 1, 1]).reshape([3, 3])
            ones_5 = np.array([1, 3, 3, 1, 0, 3, 3, 3, 3]).reshape([3, 3])
            ones_6 = np.array([3, 3, 3, 1, 0, 3, 1, 3, 3]).reshape([3, 3])
            ones_7 = np.array([3, 3, 1, 3, 0, 1, 3, 3, 3]).reshape([3, 3])
            ones_8 = np.array([3, 3, 3, 3, 0, 1, 3, 3, 1]).reshape([3, 3])

            nbeibour_1 = signal.correlate2d(edges, ones_1, boundary='fill', mode='same')
            nbeibour_2 = signal.correlate2d(edges, ones_2, boundary='fill', mode='same')
            nbeibour_3 = signal.correlate2d(edges, ones_3, boundary='fill', mode='same')
            nbeibour_4 = signal.correlate2d(edges, ones_4, boundary='fill', mode='same')
            nbeibour_5 = signal.correlate2d(edges, ones_5, boundary='fill', mode='same')
            nbeibour_6 = signal.correlate2d(edges, ones_6, boundary='fill', mode='same')
            nbeibour_7 = signal.correlate2d(edges, ones_7, boundary='fill', mode='same')
            nbeibour_8 = signal.correlate2d(edges, ones_8, boundary='fill', mode='same')
            return nbeibour_1, nbeibour_2, nbeibour_3, nbeibour_4, nbeibour_5, nbeibour_6, nbeibour_7, nbeibour_8

        edges_cp = edges.copy()
        edges_cp[np.where(edges_cp == -1)] = 100

        edges[np.where(edges != 1)] = 0
        positive_advance = -1 * np.ones(edges.shape)

        ones = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1]).reshape([3, 3])
        region = signal.correlate2d(edges, ones, boundary='fill', mode='same')
        bound = signal.correlate2d(edges_cp, np.ones([9, 9]), boundary='fill', mode='same')
        n1, n2, n3, n4, n5, n6, n7, n8 = filt_0(edges)

        positive_advance[np.where(region == 1)] = 1
        positive_advance[np.where(n1 == 2)] = 1
        positive_advance[np.where(n2 == 2)] = 1
        positive_advance[np.where(n3 == 2)] = 1
        positive_advance[np.where(n4 == 2)] = 1
        positive_advance[np.where(n5 == 2)] = 1
        positive_advance[np.where(n6 == 2)] = 1
        positive_advance[np.where(n7 == 2)] = 1
        positive_advance[np.where(n8 == 2)] = 1
        positive_advance[np.where(edges != 1)] = 0
        positive_advance[np.where(bound >= 100)] = 0

        return positive_advance

    def iso_point_shuxing(positive_advance, edges, neib=[], raugh=False):
        if raugh:
            edge_shuxing = -1 * np.ones(positive_advance.shape)
            s1, s2 = np.where(positive_advance == 1)
            for i in range(len(s1)):
                edges_cp = edges.copy()
                if len(neib) > 0:
                    edges_cp[np.where(neib == 3)] = 3  # 加入隔断

                fangxiang = np.zeros([4])
                fangxiang_num = np.zeros([4])
                x = s1[i]
                y = s2[i]
                edges_cp[x, y] = 0
                num = 0
                while (num < 15):
                    if edges_cp[x - 1, y] == 1:
                        if np.sum(fangxiang) <= 1 or fangxiang[0] == 1:
                            fangxiang[0] = 1
                            fangxiang_num[0] = fangxiang_num[0] + 1
                            x = x - 1
                            num = num + 1
                            edges_cp[x, y] = 0
                        else:
                            break
                    elif edges_cp[x, y + 1] == 1:
                        if np.sum(fangxiang) <= 1 or fangxiang[1] == 1:
                            fangxiang[1] = 1
                            fangxiang_num[1] = fangxiang_num[1] + 1
                            y = y + 1
                            num = num + 1
                            edges_cp[x, y] = 0
                        else:
                            break
                    elif edges_cp[x + 1, y] == 1:
                        if np.sum(fangxiang) <= 1 or fangxiang[2] == 1:
                            fangxiang[2] = 1
                            fangxiang_num[2] = fangxiang_num[2] + 1
                            x = x + 1
                            num = num + 1
                            edges_cp[x, y] = 0
                        else:
                            break
                    elif edges_cp[x, y - 1] == 1:
                        if np.sum(fangxiang) <= 1 or fangxiang[3] == 1:
                            fangxiang[3] = 1
                            fangxiang_num[3] = fangxiang_num[3] + 1
                            y = y - 1
                            num = num + 1
                            edges_cp[x, y] = 0
                        else:
                            break
                    else:
                        break
                if (fangxiang[0] == 1 and np.sum(fangxiang) == 1) or fangxiang_num[0] > 10:
                    edge_shuxing[s1[i], s2[i]] = 1
                elif (fangxiang[1] == 1 and np.sum(fangxiang) == 1) or fangxiang_num[1] > 10:
                    edge_shuxing[s1[i], s2[i]] = 2
                elif (fangxiang[2] == 1 and np.sum(fangxiang) == 1) or fangxiang_num[2] > 10:
                    edge_shuxing[s1[i], s2[i]] = 3
                elif (fangxiang[3] == 1 and np.sum(fangxiang) == 1) or fangxiang_num[3] > 10:
                    edge_shuxing[s1[i], s2[i]] = 4
                elif fangxiang[0] == 1 and fangxiang[1] == 1 and np.sum(fangxiang) == 2:
                    edge_shuxing[s1[i], s2[i]] = 5
                elif fangxiang[1] == 1 and fangxiang[2] == 1 and np.sum(fangxiang) == 2:
                    edge_shuxing[s1[i], s2[i]] = 6
                elif fangxiang[2] == 1 and fangxiang[3] == 1 and np.sum(fangxiang) == 2:
                    edge_shuxing[s1[i], s2[i]] = 7
                elif fangxiang[3] == 1 and fangxiang[0] == 1 and np.sum(fangxiang) == 2:
                    edge_shuxing[s1[i], s2[i]] = 8
                elif np.sum(fangxiang) == 0:
                    if edges[s1[i] - 1, s2[i] - 1] == 1:
                        edge_shuxing[s1[i], s2[i]] = 8
                    if edges[s1[i] - 1, s2[i] + 1] == 1:
                        edge_shuxing[s1[i], s2[i]] = 5
                    if edges[s1[i] + 1, s2[i] - 1] == 1:
                        edge_shuxing[s1[i], s2[i]] = 7
                    if edges[s1[i] + 1, s2[i] + 1] == 1:
                        edge_shuxing[s1[i], s2[i]] = 6
        else:
            edge_shuxing = -1 * np.ones(positive_advance.shape)
            s1, s2 = np.where(positive_advance == 1)
            for i in range(len(s1)):
                edges_cp = edges.copy()
                if len(neib) > 0:
                    edges_cp[np.where(neib == 3)] = 3  # 加入隔断

                fangxiang = np.zeros([4])
                x = s1[i]
                y = s2[i]
                edges_cp[x, y] = 0
                num = 0
                while (num < 15):
                    if edges_cp[x - 1, y] == 1:
                        if np.sum(fangxiang) <= 1 or fangxiang[0] == 1:
                            fangxiang[0] = 1
                            x = x - 1
                            num = num + 1
                            edges_cp[x, y] = 0
                        else:
                            break
                    elif edges_cp[x, y + 1] == 1:
                        if np.sum(fangxiang) <= 1 or fangxiang[1] == 1:
                            fangxiang[1] = 1
                            y = y + 1
                            num = num + 1
                            edges_cp[x, y] = 0
                        else:
                            break
                    elif edges_cp[x + 1, y] == 1:
                        if np.sum(fangxiang) <= 1 or fangxiang[2] == 1:
                            fangxiang[2] = 1
                            x = x + 1
                            num = num + 1
                            edges_cp[x, y] = 0
                        else:
                            break
                    elif edges_cp[x, y - 1] == 1:
                        if np.sum(fangxiang) <= 1 or fangxiang[3] == 1:
                            fangxiang[3] = 1
                            y = y - 1
                            num = num + 1
                            edges_cp[x, y] = 0
                        else:
                            break
                    else:
                        break
                if fangxiang[0] == 1 and np.sum(fangxiang) == 1:
                    edge_shuxing[s1[i], s2[i]] = 1
                elif fangxiang[1] == 1 and np.sum(fangxiang) == 1:
                    edge_shuxing[s1[i], s2[i]] = 2
                elif fangxiang[2] == 1 and np.sum(fangxiang) == 1:
                    edge_shuxing[s1[i], s2[i]] = 3
                elif fangxiang[3] == 1 and np.sum(fangxiang) == 1:
                    edge_shuxing[s1[i], s2[i]] = 4
                elif fangxiang[0] == 1 and fangxiang[1] == 1 and np.sum(fangxiang) == 2:
                    edge_shuxing[s1[i], s2[i]] = 5
                elif fangxiang[1] == 1 and fangxiang[2] == 1 and np.sum(fangxiang) == 2:
                    edge_shuxing[s1[i], s2[i]] = 6
                elif fangxiang[2] == 1 and fangxiang[3] == 1 and np.sum(fangxiang) == 2:
                    edge_shuxing[s1[i], s2[i]] = 7
                elif fangxiang[3] == 1 and fangxiang[0] == 1 and np.sum(fangxiang) == 2:
                    edge_shuxing[s1[i], s2[i]] = 8
                elif np.sum(fangxiang) == 0:
                    if edges[s1[i] - 1, s2[i] - 1] == 1:
                        edge_shuxing[s1[i], s2[i]] = 8
                    if edges[s1[i] - 1, s2[i] + 1] == 1:
                        edge_shuxing[s1[i], s2[i]] = 5
                    if edges[s1[i] + 1, s2[i] - 1] == 1:
                        edge_shuxing[s1[i], s2[i]] = 7
                    if edges[s1[i] + 1, s2[i] + 1] == 1:
                        edge_shuxing[s1[i], s2[i]] = 6

        return edge_shuxing

    def ios_point_shuxing_mask(edge, mask_x, mask_y):
        fangxiang = np.zeros([4])
        fangxiang_num = np.zeros([4])
        edges_cp = edge.copy()
        x = mask_x
        y = mask_y
        edges_cp[x, y] = 0
        num = 0
        shuxing = 0
        while (num < 15):
            if edges_cp[x - 1, y] == 1:
                if np.sum(fangxiang) <= 1 or fangxiang[0] == 1:
                    fangxiang[0] = 1
                    fangxiang_num[0] = fangxiang_num[0] + 1
                    x = x - 1
                    num = num + 1
                    edges_cp[x, y] = 0
                else:
                    break
            elif edges_cp[x, y + 1] == 1:
                if np.sum(fangxiang) <= 1 or fangxiang[1] == 1:
                    fangxiang[1] = 1
                    fangxiang_num[1] = fangxiang_num[1] + 1
                    y = y + 1
                    num = num + 1
                    edges_cp[x, y] = 0
                else:
                    break
            elif edges_cp[x + 1, y] == 1:
                if np.sum(fangxiang) <= 1 or fangxiang[2] == 1:
                    fangxiang[2] = 1
                    fangxiang_num[2] = fangxiang_num[2] + 1
                    x = x + 1
                    num = num + 1
                    edges_cp[x, y] = 0
                else:
                    break
            elif edges_cp[x, y - 1] == 1:
                if np.sum(fangxiang) <= 1 or fangxiang[3] == 1:
                    fangxiang[3] = 1
                    fangxiang_num[3] = fangxiang_num[3] + 1
                    y = y - 1
                    num = num + 1
                    edges_cp[x, y] = 0
                else:
                    break
            else:
                break
        if (fangxiang[0] == 1 and np.sum(fangxiang) == 1) or fangxiang_num[0] > 10:
            shuxing = 1
        elif (fangxiang[1] == 1 and np.sum(fangxiang) == 1) or fangxiang_num[1] > 10:
            shuxing = 2
        elif (fangxiang[2] == 1 and np.sum(fangxiang) == 1) or fangxiang_num[2] > 10:
            shuxing = 3
        elif (fangxiang[3] == 1 and np.sum(fangxiang) == 1) or fangxiang_num[3] > 10:
            shuxing = 4
        elif fangxiang[0] == 1 and fangxiang[1] == 1 and np.sum(fangxiang) == 2:
            shuxing = 5
        elif fangxiang[1] == 1 and fangxiang[2] == 1 and np.sum(fangxiang) == 2:
            shuxing = 6
        elif fangxiang[2] == 1 and fangxiang[3] == 1 and np.sum(fangxiang) == 2:
            shuxing = 7
        elif fangxiang[3] == 1 and fangxiang[0] == 1 and np.sum(fangxiang) == 2:
            shuxing = 8
        return shuxing

    def con_mask(edges, x_a, y_b, mask, distance, shuxing_mask, weight_matrix, conn_mask_valu):
        edges_pad = edges
        x = x_a
        y = y_b
        x_ = 0
        y_ = 0
        flag = False
        con_value = conn_mask_valu

        if shuxing_mask == 1:
            weight_tmp = weight_matrix[x:x + distance + 1, y - distance:y + distance + 1]
            mask_tmp = mask[x:x + distance + 1, y - distance:y + distance + 1]
            weight_tmp_h, weight_tmp_w = mask_tmp.shape
            xx = np.dstack(np.unravel_index(np.argsort(-weight_tmp.ravel()), (weight_tmp_h, weight_tmp_w)))[0]
            x_tmp = xx[:,0]
            y_tmp = xx[:,1]
            for idx_tmp in range(len(x_tmp)):
                if weight_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] <= 0:
                    break
                if (x_tmp[idx_tmp] - 0) ** 2 +  (y_tmp[idx_tmp] - distance) ** 2 > distance ** 2:
                    continue
                if mask_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] == 255:
                    continue

                x_ = x_tmp[idx_tmp] + x - 0
                y_ = y_tmp[idx_tmp] + y - distance

                min_x = min(x, x_)
                min_y = min(y, y_)
                max_x = max(x, x_) + 1
                max_y = max(y, y_) + 1
                if x_ > x and y_ > y:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=True)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                else:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=False)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                flag = True
                break


        elif shuxing_mask == 2:
            weight_tmp = weight_matrix[x - distance:x + distance + 1, y - distance:y + 1]
            mask_tmp = mask[x - distance:x + distance + 1, y - distance:y + 1]
            weight_tmp_h, weight_tmp_w = mask_tmp.shape
            xx = np.dstack(np.unravel_index(np.argsort(-weight_tmp.ravel()), (weight_tmp_h, weight_tmp_w)))[0]
            x_tmp = xx[:,0]
            y_tmp = xx[:,1]
            for idx_tmp in range(len(x_tmp)):
                if weight_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] <= 0:
                    break
                if (x_tmp[idx_tmp] - distance) ** 2 +  (y_tmp[idx_tmp] - distance) ** 2 > distance ** 2:
                    continue
                if mask_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] == 255:
                    continue

                x_ = x_tmp[idx_tmp] + x - distance
                y_ = y_tmp[idx_tmp] + y - distance

                min_x = min(x, x_)
                min_y = min(y, y_)
                max_x = max(x, x_) + 1
                max_y = max(y, y_) + 1
                if x_ > x and y_ > y:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=True)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                else:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=False)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                flag = True
                break

        elif shuxing_mask == 3:
            weight_tmp = weight_matrix[x - distance:x + 1, y - distance:y + distance + 1]
            mask_tmp = mask[x - distance:x + 1, y - distance:y + distance + 1]
            weight_tmp_h, weight_tmp_w = mask_tmp.shape
            xx = np.dstack(np.unravel_index(np.argsort(-weight_tmp.ravel()), (weight_tmp_h, weight_tmp_w)))[0]
            x_tmp = xx[:,0]
            y_tmp = xx[:,1]
            for idx_tmp in range(len(x_tmp)):
                if weight_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] <= 0:
                    break
                if (x_tmp[idx_tmp] - distance) ** 2 +  (y_tmp[idx_tmp] - distance) ** 2 > distance ** 2:
                    continue
                if mask_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] == 255:
                    continue

                x_ = x_tmp[idx_tmp] + x - distance
                y_ = y_tmp[idx_tmp] + y - distance

                min_x = min(x, x_)
                min_y = min(y, y_)
                max_x = max(x, x_) + 1
                max_y = max(y, y_) + 1
                if x_ > x and y_ > y:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=True)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                else:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=False)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                flag = True
                break


        elif shuxing_mask == 4:
            weight_tmp = weight_matrix[x - distance:x + distance + 1, y:y + distance + 1]
            mask_tmp = mask[x - distance:x + distance + 1, y:y + distance + 1]
            weight_tmp_h, weight_tmp_w = mask_tmp.shape
            xx = np.dstack(np.unravel_index(np.argsort(-weight_tmp.ravel()), (weight_tmp_h, weight_tmp_w)))[0]
            x_tmp = xx[:,0]
            y_tmp = xx[:,1]
            for idx_tmp in range(len(x_tmp)):
                if weight_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] <= 0:
                    break
                if (x_tmp[idx_tmp] - distance) ** 2 +  (y_tmp[idx_tmp] - 0) ** 2 > distance ** 2:
                    continue
                if mask_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] == 255:
                    continue

                x_ = x_tmp[idx_tmp] + x - distance
                y_ = y_tmp[idx_tmp] + y - 0

                min_x = min(x, x_)
                min_y = min(y, y_)
                max_x = max(x, x_) + 1
                max_y = max(y, y_) + 1
                if x_ > x and y_ > y:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=True)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                else:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=False)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                flag = True
                break


        elif shuxing_mask == 5:
            weight_tmp = weight_matrix[x:x + distance + 1, y - distance:y + 1]
            mask_tmp = mask[x:x + distance + 1, y - distance:y + 1]
            weight_tmp_h, weight_tmp_w = mask_tmp.shape
            xx = np.dstack(np.unravel_index(np.argsort(-weight_tmp.ravel()), (weight_tmp_h, weight_tmp_w)))[0]
            x_tmp = xx[:,0]
            y_tmp = xx[:,1]
            for idx_tmp in range(len(x_tmp)):
                if weight_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] <= 0:
                    break
                if (x_tmp[idx_tmp] - 0) ** 2 +  (y_tmp[idx_tmp] - distance) ** 2 > distance ** 2:
                    continue
                if mask_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] == 255:
                    continue

                x_ = x_tmp[idx_tmp] + x - 0
                y_ = y_tmp[idx_tmp] + y - distance

                min_x = min(x, x_)
                min_y = min(y, y_)
                max_x = max(x, x_) + 1
                max_y = max(y, y_) + 1
                if x_ > x and y_ > y:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=True)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                else:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=False)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                flag = True
                break


        elif shuxing_mask == 6:
            weight_tmp = weight_matrix[x - distance:x + 1, y - distance:y + 1]
            mask_tmp = mask[x - distance:x + 1, y - distance:y + 1]
            weight_tmp_h, weight_tmp_w = mask_tmp.shape
            xx = np.dstack(np.unravel_index(np.argsort(-weight_tmp.ravel()), (weight_tmp_h, weight_tmp_w)))[0]
            x_tmp = xx[:,0]
            y_tmp = xx[:,1]
            for idx_tmp in range(len(x_tmp)):
                if weight_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] <= 0:
                    break
                if (x_tmp[idx_tmp] - distance) ** 2 + (y_tmp[idx_tmp] - distance) ** 2 > distance ** 2:
                    continue
                if mask_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] == 255:
                    continue

                x_ = x_tmp[idx_tmp] + x - distance
                y_ = y_tmp[idx_tmp] + y - distance

                min_x = min(x, x_)
                min_y = min(y, y_)
                max_x = max(x, x_) + 1
                max_y = max(y, y_) + 1
                if x_ > x and y_ > y:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=True)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                else:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=False)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                flag = True
                break

        elif shuxing_mask == 7:
            weight_tmp = weight_matrix[x - distance:x + 1, y:y + distance + 1]
            mask_tmp = mask[x - distance:x + 1, y:y + distance + 1]
            weight_tmp_h, weight_tmp_w = mask_tmp.shape
            xx = np.dstack(np.unravel_index(np.argsort(-weight_tmp.ravel()), (weight_tmp_h, weight_tmp_w)))[0]
            x_tmp = xx[:,0]
            y_tmp = xx[:,1]
            for idx_tmp in range(len(x_tmp)):
                if weight_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] <= 0:
                    break
                if (x_tmp[idx_tmp] - distance) ** 2 +  (y_tmp[idx_tmp] - 0) ** 2 > distance ** 2:
                    continue
                if mask_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] == 255:
                    continue

                x_ = x_tmp[idx_tmp] + x - distance
                y_ = y_tmp[idx_tmp] + y - 0

                min_x = min(x, x_)
                min_y = min(y, y_)
                max_x = max(x, x_) + 1
                max_y = max(y, y_) + 1
                if x_ > x and y_ > y:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=True)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                else:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=False)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                flag = True
                break

        elif shuxing_mask == 8:
            weight_tmp = weight_matrix[x:x + distance + 1, y:y + distance + 1]
            mask_tmp = mask[x:x + distance + 1, y:y + distance + 1]
            weight_tmp_h, weight_tmp_w = mask_tmp.shape
            xx = np.dstack(np.unravel_index(np.argsort(-weight_tmp.ravel()), (weight_tmp_h, weight_tmp_w)))[0]
            x_tmp = xx[:,0]
            y_tmp = xx[:,1]
            for idx_tmp in range(len(x_tmp)):
                if weight_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] <= 0:
                    break
                if (x_tmp[idx_tmp] - 0) ** 2 +  (y_tmp[idx_tmp] - 0) ** 2 > distance ** 2:
                    continue
                if mask_tmp[x_tmp[idx_tmp], y_tmp[idx_tmp]] == 255:
                    continue

                x_ = x_tmp[idx_tmp] + x - 0
                y_ = y_tmp[idx_tmp] + y - 0

                min_x = min(x, x_)
                min_y = min(y, y_)
                max_x = max(x, x_) + 1
                max_y = max(y, y_) + 1
                if x_ > x and y_ > y:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=True)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                else:
                    route, route_score = get_route(weight_matrix[min_x:max_x, min_y:max_y], x_y_same=False)
                    edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                flag = True
                break

        return edges, x_, y_, flag

    def connection(edges, distance=10, weight_matrix = []):
        con_value = 1.01
        iso_point_map = iso_point(edges.copy())
        edge_shuxing = iso_point_shuxing(iso_point_map, edges.copy())

        edge_shuxing_pad = -1 * np.ones([edge_shuxing.shape[0] + distance * 2, edge_shuxing.shape[1] + distance * 2])

        edges_pad = 0 * np.ones([edges.shape[0] + distance * 2, edges.shape[1] + distance * 2])
        weight_matrix_pad = 0 * np.ones([edges.shape[0] + distance * 2, edges.shape[1] + distance * 2])

        edge_shuxing_pad[distance:distance + edge_shuxing.shape[0],
        distance:distance + edge_shuxing.shape[1]] = edge_shuxing
        edges_pad[distance:distance + edges.shape[0], distance:distance + edges.shape[1]] = edges
        weight_matrix_pad[distance:distance + edges.shape[0], distance:distance + edges.shape[1]] = weight_matrix
        edges_pad_cp = edges_pad.copy()

        s1, s2 = np.where(edge_shuxing_pad > 0)
        for i in range(s1.shape[0]):

            x = s1[i]
            y = s2[i]

            if x == 134:
                print('ss')

            if edge_shuxing_pad[x, y] == 0:
                edges_pad[x, y] = 0
                e1, e2 = np.where(edges_pad[x - distance:x + distance + 1, y - distance:y + distance + 1] == 1)
                if len(e1) > 0:

                    dis_jihe = min_dis_getjihe(e1, e2, distance, distance)
                    dis_idx_jihe = np.argsort(dis_jihe)
                    for dis_idx in dis_idx_jihe:
                        if dis_jihe[dis_idx] <= distance:
                            x_ = e1[dis_idx] + x - distance
                            y_ = e2[dis_idx] + y - distance
                            min_x = min(x,x_)
                            min_y = min(y,y_)
                            max_x = max(x,x_)+1
                            max_y = max(y,y_)+1
                            if x_ > x and y_ > y:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=True)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value
                            else:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=False)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value

                            edges_pad[x_, y_] = 1
                            pos_num_connect = len(np.where(np.logical_and(edges_pad[min_x:max_x, min_y:max_y] == 1, route == 1) == True)[0])
                            neg_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == 0)[0].shape[0]
                            fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edges_pad == con_value)] == muhu_value)[0].shape[0]
                            if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio or route_score <= 0 or pos_num_connect > 1:
                                edges_pad[np.where(edges_pad == con_value)] = edges_pad_cp[
                                    np.where(edges_pad == con_value)]
                            else:
                                edges_pad[np.where(edges_pad == con_value)] = 1
                                break
                            edges_pad[x, y] = 0
                        else:
                            break
                edges_pad[x, y] = 1

            elif edge_shuxing_pad[x, y] == 1:
                edges_pad[x, y] = 0
                e1, e2 = np.where(edges_pad[x:x + distance + 1, y - distance:y + distance + 1] == 1)
                if len(e1) > 0:

                    dis_jihe = min_dis_getjihe(e1, e2, 0, distance)
                    dis_idx_jihe = np.argsort(dis_jihe)
                    for dis_idx in dis_idx_jihe:
                        if dis_jihe[dis_idx] <= distance:

                            x_ = e1[dis_idx] + x - 0
                            y_ = e2[dis_idx] + y - distance

                            min_x = min(x,x_)
                            min_y = min(y,y_)
                            max_x = max(x,x_)+1
                            max_y = max(y,y_)+1
                            if x_ > x and y_ > y:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=True)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value
                            else:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=False)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value


                            edges_pad[x_, y_] = 1
                            pos_num_connect = len(np.where(np.logical_and(edges_pad[min_x:max_x, min_y:max_y] == 1, route == 1) == True)[0])
                            neg_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == 0)[0].shape[0]
                            fuz_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == muhu_value)[0].shape[0]

                            if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio or route_score <= 0 or pos_num_connect > 1:
                                edges_pad[np.where(edges_pad == con_value)] = edges_pad_cp[np.where(edges_pad == con_value)]
                            else:
                                edges_pad[np.where(edges_pad == con_value)] = 1
                                break
                            edges_pad[x, y] = 0
                        else:
                            break
                edges_pad[x, y] = 1

            elif edge_shuxing_pad[x, y] == 2:
                edges_pad[x, y] = 0

                e1, e2 = np.where(edges_pad[x - distance:x + distance + 1, y - distance:y + 1] == 1)
                if len(e1) > 0:
                    dis_jihe = min_dis_getjihe(e1, e2, distance, distance)
                    dis_idx_jihe = np.argsort(dis_jihe)
                    for dis_idx in dis_idx_jihe:
                        if dis_jihe[dis_idx] <= distance:
                            x_ = e1[dis_idx] + x - distance
                            y_ = e2[dis_idx] + y - distance


                            min_x = min(x,x_)
                            min_y = min(y,y_)
                            max_x = max(x,x_)+1
                            max_y = max(y,y_)+1
                            if x_ > x and y_ > y:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=True)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value
                            else:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=False)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value


                            edges_pad[x_, y_] = 1
                            pos_num_connect = len(np.where(np.logical_and(edges_pad[min_x:max_x, min_y:max_y] == 1, route == 1) == True)[0])
                            neg_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == 0)[0].shape[0]
                            fuz_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == muhu_value)[0].shape[0]
                            if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio or route_score <= 0 or pos_num_connect > 1:
                                edges_pad[np.where(edges_pad == con_value)] = edges_pad_cp[
                                    np.where(edges_pad == con_value)]
                            else:
                                edges_pad[np.where(edges_pad == con_value)] = 1
                                break
                            edges_pad[x, y] = 0
                        else:
                            break
                edges_pad[x, y] = 1

            elif edge_shuxing_pad[x, y] == 3:
                edges_pad[x, y] = 0

                e1, e2 = np.where(edges_pad[x - distance:x + 1, y - distance:y + distance + 1] == 1)
                if len(e1) > 0:

                    dis_jihe = min_dis_getjihe(e1, e2, distance, distance)
                    dis_idx_jihe = np.argsort(dis_jihe)
                    for dis_idx in dis_idx_jihe:
                        if dis_jihe[dis_idx] <= distance:
                            x_ = e1[dis_idx] + x - distance
                            y_ = e2[dis_idx] + y - distance

                            min_x = min(x,x_)
                            min_y = min(y,y_)
                            max_x = max(x,x_)+1
                            max_y = max(y,y_)+1
                            if x_ > x and y_ > y:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=True)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value
                            else:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=False)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value

                            edges_pad[x_, y_] = 1
                            pos_num_connect = len(np.where(np.logical_and(edges_pad[min_x:max_x, min_y:max_y] == 1, route == 1) == True)[0])
                            neg_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == 0)[0].shape[0]
                            fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edges_pad == con_value)] == muhu_value)[0].shape[0]
                            if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio  or route_score <= 0 or pos_num_connect > 1:
                                edges_pad[np.where(edges_pad == con_value)] = edges_pad_cp[np.where(edges_pad == con_value)]
                            else:
                                edges_pad[np.where(edges_pad == con_value)] = 1
                                break
                            edges_pad[x, y] = 0
                        else:
                            break
                edges_pad[x, y] = 1

            elif edge_shuxing_pad[x, y] == 4:
                edges_pad[x, y] = 0

                e1, e2 = np.where(edges_pad[x - distance:x + distance + 1, y:y + distance + 1] == 1)
                if len(e1) > 0:

                    dis_jihe = min_dis_getjihe(e1, e2, distance, 0)
                    dis_idx_jihe = np.argsort(dis_jihe)
                    for dis_idx in dis_idx_jihe:
                        if dis_jihe[dis_idx] <= distance:
                            x_ = e1[dis_idx] + x - distance
                            y_ = e2[dis_idx] + y - 0

                            min_x = min(x,x_)
                            min_y = min(y,y_)
                            max_x = max(x,x_)+1
                            max_y = max(y,y_)+1
                            if x_ > x and y_ > y:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=True)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value
                            else:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x,min_y:max_y], x_y_same=False)
                                edges_pad[min_x:max_x,min_y:max_y][np.logical_and(edges_pad[min_x:max_x,min_y:max_y]!=1, route == 1)] = con_value

                            edges_pad[x_, y_] = 1
                            pos_num_connect = len(np.where(np.logical_and(edges_pad[min_x:max_x, min_y:max_y] == 1, route == 1) == True)[0])
                            neg_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == 0)[0].shape[0]
                            fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edges_pad == con_value)] == muhu_value)[0].shape[0]
                            if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio or route_score <= 0 or pos_num_connect > 1:
                                edges_pad[np.where(edges_pad == con_value)] = edges_pad_cp[
                                    np.where(edges_pad == con_value)]
                            else:
                                edges_pad[np.where(edges_pad == con_value)] = 1
                                break
                            edges_pad[x, y] = 0
                        else:
                            break
                edges_pad[x, y] = 1

            elif edge_shuxing_pad[x, y] == 5:
                edges_pad[x, y] = 0

                e1, e2 = np.where(edges_pad[x:x + distance + 1, y - distance:y + 1] == 1)
                if len(e1) > 0:
                    dis_jihe = min_dis_getjihe(e1, e2, 0, distance)
                    dis_idx_jihe = np.argsort(dis_jihe)
                    for dis_idx in dis_idx_jihe:
                        if dis_jihe[dis_idx] <= distance:
                            x_ = e1[dis_idx] + x - 0
                            y_ = e2[dis_idx] + y - distance

                            min_x = min(x, x_)
                            min_y = min(y, y_)
                            max_x = max(x, x_) + 1
                            max_y = max(y, y_) + 1
                            if x_ > x and y_ > y:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x, min_y:max_y], x_y_same=True)
                                edges_pad[min_x:max_x, min_y:max_y][
                                    np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1,
                                                   route == 1)] = con_value
                            else:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x, min_y:max_y], x_y_same=False)
                                edges_pad[min_x:max_x, min_y:max_y][
                                    np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1,
                                                   route == 1)] = con_value

                            edges_pad[x_, y_] = 1
                            pos_num_connect = len(np.where(np.logical_and(edges_pad[min_x:max_x, min_y:max_y] == 1, route == 1) == True)[0])
                            neg_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == 0)[0].shape[0]
                            fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edges_pad == con_value)] == muhu_value)[0].shape[0]
                            if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio  or route_score <= 0 or pos_num_connect > 1:
                                edges_pad[np.where(edges_pad == con_value)] = edges_pad_cp[
                                    np.where(edges_pad == con_value)]
                            else:
                                edges_pad[np.where(edges_pad == con_value)] = 1
                                break
                            edges_pad[x, y] = 0
                        else:
                            break
                edges_pad[x, y] = 1

            elif edge_shuxing_pad[x, y] == 6:
                edges_pad[x, y] = 0

                e1, e2 = np.where(edges_pad[x - distance:x + 1, y - distance:y + 1] == 1)
                if len(e1) > 0:
                    dis_jihe = min_dis_getjihe(e1, e2, distance, distance)
                    dis_idx_jihe = np.argsort(dis_jihe)
                    for dis_idx in dis_idx_jihe:
                        if dis_jihe[dis_idx] <= distance:
                            x_ = e1[dis_idx] + x - distance
                            y_ = e2[dis_idx] + y - distance

                            min_x = min(x, x_)
                            min_y = min(y, y_)
                            max_x = max(x, x_) + 1
                            max_y = max(y, y_) + 1
                            if x_ > x and y_ > y:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x, min_y:max_y], x_y_same=True)
                                edges_pad[min_x:max_x, min_y:max_y][
                                    np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1,
                                                   route == 1)] = con_value
                            else:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x, min_y:max_y], x_y_same=False)
                                edges_pad[min_x:max_x, min_y:max_y][
                                    np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1,
                                                   route == 1)] = con_value

                            edges_pad[x_, y_] = 1
                            pos_num_connect = len(np.where(np.logical_and(edges_pad[min_x:max_x, min_y:max_y] == 1, route == 1) == True)[0])
                            neg_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == 0)[0].shape[0]
                            fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edges_pad == con_value)] == muhu_value)[0].shape[0]
                            if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio   or route_score <= 0 or pos_num_connect > 1:
                                edges_pad[np.where(edges_pad == con_value)] = edges_pad_cp[
                                    np.where(edges_pad == con_value)]
                            else:
                                edges_pad[np.where(edges_pad == con_value)] = 1
                                break
                            edges_pad[x, y] = 0
                        else:
                            break
                edges_pad[x, y] = 1

            elif edge_shuxing_pad[x, y] == 7:
                edges_pad[x, y] = 0

                e1, e2 = np.where(edges_pad[x - distance:x + 1, y:y + distance + 1] == 1)
                if len(e1) > 0:

                    dis_jihe = min_dis_getjihe(e1, e2, distance, 0)
                    dis_idx_jihe = np.argsort(dis_jihe)
                    for dis_idx in dis_idx_jihe:
                        if dis_jihe[dis_idx] <= distance:
                            x_ = e1[dis_idx] + x - distance
                            y_ = e2[dis_idx] + y - 0

                            if x_ == 176 and y_ == 268:
                                print('ss')

                            min_x = min(x, x_)
                            min_y = min(y, y_)
                            max_x = max(x, x_) + 1
                            max_y = max(y, y_) + 1

                            if x_ > x and y_ > y:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x, min_y:max_y], x_y_same=True)
                                edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value
                            else:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x, min_y:max_y], x_y_same=False)
                                edges_pad[min_x:max_x, min_y:max_y][np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1, route == 1)] = con_value

                            edges_pad[x_, y_] = 1
                            pos_num_connect = len(np.where(np.logical_and(edges_pad[min_x:max_x, min_y:max_y] == 1, route == 1) == True)[0])
                            neg_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == 0)[0].shape[0]
                            fuz_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == muhu_value)[0].shape[0]
                            if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio or route_score <= 0 or pos_num_connect > 1:
                                edges_pad[np.where(edges_pad == con_value)] = edges_pad_cp[
                                    np.where(edges_pad == con_value)]
                            else:
                                edges_pad[np.where(edges_pad == con_value)] = 1
                                break
                            edges_pad[x, y] = 0
                        else:
                            break
                edges_pad[x, y] = 1

            elif edge_shuxing_pad[x, y] == 8:
                edges_pad[x, y] = 0

                e1, e2 = np.where(edges_pad[x:x + distance + 1, y:y + distance + 1] == 1)
                if len(e1) > 0:

                    dis_jihe = min_dis_getjihe(e1, e2, 0, 0)
                    dis_idx_jihe = np.argsort(dis_jihe)
                    for dis_idx in dis_idx_jihe:
                        if dis_jihe[dis_idx] <= distance:
                            x_ = e1[dis_idx] + x - 0
                            y_ = e2[dis_idx] + y - 0

                            min_x = min(x, x_)
                            min_y = min(y, y_)
                            max_x = max(x, x_) + 1
                            max_y = max(y, y_) + 1
                            if x_ > x and y_ > y:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x, min_y:max_y], x_y_same=True)
                                edges_pad[min_x:max_x, min_y:max_y][
                                    np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1,
                                                   route == 1)] = con_value
                            else:
                                route, route_score = get_route(weight_matrix_pad[min_x:max_x, min_y:max_y], x_y_same=False)
                                edges_pad[min_x:max_x, min_y:max_y][
                                    np.logical_and(edges_pad[min_x:max_x, min_y:max_y] != 1,
                                                   route == 1)] = con_value
                            edges_pad[x_, y_] = 1
                            pos_num_connect = len(np.where(np.logical_and(edges_pad[min_x:max_x, min_y:max_y] == 1, route == 1) == True)[0])
                            neg_num_connect = np.where(edges_pad_cp[np.where(edges_pad == con_value)] == 0)[0].shape[0]
                            fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edges_pad == con_value)] == muhu_value)[0].shape[0]
                            if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio or route_score <= 0 or pos_num_connect > 1:
                                edges_pad[np.where(edges_pad == con_value)] = edges_pad_cp[
                                    np.where(edges_pad == con_value)]
                            else:
                                edges_pad[np.where(edges_pad == con_value)] = 1
                                break
                            edges_pad[x, y] = 0
                        else:
                            break
                edges_pad[x, y] = 1

        edges = edges_pad[distance:distance + edges.shape[0], distance:distance + edges.shape[1]]
        return edges

    def dele_hole(edge):
        def getGrayDiff(img, currentPoint, tmpPoint):
            return img[currentPoint.x, currentPoint.y] * img[tmpPoint.x, tmpPoint.y]

        class Point(object):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def getX(self):
                return self.x

            def getY(self):
                return self.y

        def selectConnects(p):
            if p == 2:
                connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                            Point(-1, 1), Point(-1, 0), Point(-2, 0), Point(2, 0), Point(0, -2), Point(0, 2)]
            elif p == 0:
                connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                            Point(-1, 1), Point(-1, 0)]
            else:
                connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
            return connects

        def regionGrow(img, seed, img_raw, label, p=1):
            height, weight = img.shape
            seedMark = np.zeros(img.shape)
            seedList = []
            seedList.append(seed)
            connects = selectConnects(p)
            while (len(seedList) > 0):
                currentPoint = seedList.pop(0)
                seedMark[currentPoint.x, currentPoint.y] = label
                num = 4
                for i in range(num):
                    tmpX = currentPoint.x + connects[i].x
                    tmpY = currentPoint.y + connects[i].y
                    if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                        continue
                    grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                    if grayDiff != muhu_value and seedMark[tmpX, tmpY] == 0:
                        seedMark[tmpX, tmpY] = label
                        seedList.append(Point(tmpX, tmpY))

            s1, s2 = np.where(seedMark == label)
            if 0 in s1 or 0 in s2 or height - 1 in s1 or weight - 1 in s2:
                return seedMark, False

            img_tmp = img_raw[s1, s2]
            back_num = np.where(img_tmp == 0)[0].shape[0]
            muhu_num = np.where(img_tmp == muhu_value)[0].shape[0]
            # if back_num == 0 and muhu_num > 0:  这个应该去掉
            #    return seedMark, True
            if back_num + muhu_num <= region_min:
                return seedMark, True
            if back_num + muhu_num <= region_min_second and muhu_num / (muhu_num + back_num) > region_min_ratio:
                return seedMark, True

            return seedMark, False

        def hebing(edge, seedmark, lab):
            ones = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape([3, 3])
            edge_ = edge.copy()
            edge_dele = signal.convolve2d(seedmark, ones, boundary='fill', mode='same')
            edge_dele[np.where(seedmark == lab)] = 0
            edge_dele[np.where(edge != 1)] = 0
            edge_dele[np.where(edge_dele > 0)] = 1
            edge[edge_dele == 1] = muhu_value

            s1 = np.where(edge_dele > 0)[0]
            s2 = np.where(edge_dele > 0)[1]
            ss = np.vstack([s1, s2]).T
            y_min = min(ss[:, 1])
            y_max = max(ss[:, 1])
            x_min = min(ss[:, 0])
            x_max = max(ss[:, 0])
            if y_max - y_min >= x_max - x_min:
                strat_point_x = s1[np.where(s2 == y_min)]
                lab = False
                for i in strat_point_x:
                    if edge[i - 1, y_min] == 1 or edge[i + 1, y_min] == 1 or edge[i, y_min - 1] == 1 or edge[
                        i, y_min + 1] == 1:
                        edge[i, y_min] = con_value
                        x_start = i
                        lab = True
                if not lab:
                    edge[strat_point_x[0], y_min] = con_value
                    x_start = strat_point_x[0]
                for i in range(1, y_max - y_min):
                    x_tmp_min = np.min(s1[np.where(s2 == y_min + i)])
                    x_tmp_max = np.max(s1[np.where(s2 == y_min + i)])
                    x_tmp = int((x_tmp_max + x_tmp_min) / 2)
                    edge[x_tmp, y_min + i] = con_value
                    edge[min(x_tmp, x_start):max(x_tmp, x_start), y_min + i] = con_value
                    x_start = x_tmp
                end_point_x = s1[np.where(s2 == y_max)]
                lab = False
                for i in end_point_x:
                    if edge[i - 1, y_max] == 1 or edge[i + 1, y_max] == 1 or edge[i, y_max - 1] == 1 or edge[
                        i, y_max + 1] == 1:
                        edge[i, y_max] = con_value
                        x_end = i
                        lab = True
                if not lab:
                    edge[end_point_x[0], y_max] = con_value
                    x_end = end_point_x[0]
                edge[min(x_end, x_start):max(x_end, x_start), y_max] = con_value
            else:
                strat_point_y = s2[np.where(s1 == x_min)]
                lab = False
                for i in strat_point_y:
                    if edge[x_min, i - 1] == 1 or edge[x_min, i + 1] == 1 or edge[x_min - 1, i] == 1 or edge[
                        x_min + 1, i] == 1:
                        edge[x_min, i] = con_value
                        y_start = i
                        lab = True
                if not lab:
                    edge[x_min, strat_point_y[0]] = con_value
                    y_start = strat_point_y[0]

                for i in range(1, x_max - x_min):
                    y_tmp_min = np.min(s2[np.where(s1 == x_min + i)])
                    y_tmp_max = np.max(s2[np.where(s1 == x_min + i)])
                    y_tmp = int((y_tmp_max + y_tmp_min) / 2)
                    edge[x_min + i, y_tmp] = con_value
                    edge[x_min + i, min(y_tmp, y_start):max(y_tmp, y_start)] = con_value
                    y_start = y_tmp
                end_point_y = s2[np.where(s1 == x_max)]
                lab = False
                for i in end_point_y:
                    if edge[x_max, i - 1] == 1 or edge[x_max, i + 1] == 1 or edge[x_max - 1, i] == 1 or edge[
                        x_max + 1, i] == 1:
                        edge[x_max, i] = con_value
                        y_end = i
                        lab = True
                if not lab:
                    edge[x_max, end_point_y[0]] = con_value
                    y_end = end_point_y[0]
                edge[x_max, min(y_end, y_start):max(y_end, y_start)] = con_value

            return edge

        edge_cp = edge.copy()
        edge_cp_ = edge.copy()
        edge_cp[np.where(edge_cp == 0)] = muhu_value
        s1, s2 = np.where(edge_cp == muhu_value)
        seedmark_record = -1 * np.ones(edge.shape)

        needing_dele = []
        lab = 5
        for i in range(s1.shape[0]):
            x = s1[i]
            y = s2[i]
            if x == 375 and y == 348:
                print('ss')
            if seedmark_record[x, y] == -1 and edge[x, y] != 1:
                seed = Point(x, y)
                edge_cp = edge.copy()
                edge_cp[np.where(edge_cp == 0)] = muhu_value
                seedMark, mark = regionGrow(edge_cp, seed, edge, label=lab)
                if mark:
                    needing_dele.append(lab)
                    edge = hebing(edge, seedMark, lab)
                seedmark_record[seedMark == lab] = lab
                lab = lab + 1

        return edge

    def dele_cornor(edge):

        fill_value = muhu_value

        edge_cp = edge.copy()
        edge_cp[np.where(edge_cp != 1)] = 0
        ones_0 = np.array([1, 1, 1, 1, 10, 1, 1, 1, 1]).reshape([3, 3])
        nbeibour_0 = signal.correlate2d(edge_cp, ones_0, boundary='fill', mode='same')
        edge[np.where(nbeibour_0 == 10)] = fill_value
        edge_cp = edge.copy()
        edge_cp[np.where(edge_cp != 1)] = 0

        ones_5 = np.array([5, 5, 5, 0, 1, 0, 1, 1, 1]).reshape([3, 3])
        ones_6 = np.array([5, 0, 1, 5, 1, 1, 5, 0, 1]).reshape([3, 3])
        ones_7 = np.array([1, 1, 1, 0, 1, 0, 5, 5, 5]).reshape([3, 3])
        ones_8 = np.array([1, 0, 5, 1, 1, 5, 1, 0, 5]).reshape([3, 3])

        nbeibour_5 = signal.correlate2d(edge_cp, ones_5, boundary='fill', mode='same')
        edge[np.where(nbeibour_5 == 4)] = fill_value
        edge_cp = edge.copy()
        edge_cp[np.where(edge_cp != 1)] = 0
        nbeibour_6 = signal.correlate2d(edge_cp, ones_6, boundary='fill', mode='same')
        edge[np.where(nbeibour_6 == 4)] = fill_value
        edge_cp = edge.copy()
        edge_cp[np.where(edge_cp != 1)] = 0
        nbeibour_7 = signal.correlate2d(edge_cp, ones_7, boundary='fill', mode='same')
        edge[np.where(nbeibour_7 == 4)] = fill_value
        edge_cp = edge.copy()
        edge_cp[np.where(edge_cp != 1)] = 0
        nbeibour_8 = signal.correlate2d(edge_cp, ones_8, boundary='fill', mode='same')
        edge[np.where(nbeibour_8 == 4)] = fill_value

        edge_cp = edge.copy()
        edge_cp[np.where(edge_cp != 1)] = 0

        ones_1 = np.array([5, 5, 5, 5, 1, 1, 5, 1, 1]).reshape([3, 3])
        ones_2 = np.array([5, 5, 5, 1, 1, 5, 1, 1, 5]).reshape([3, 3])
        ones_3 = np.array([5, 1, 1, 5, 1, 1, 5, 5, 5]).reshape([3, 3])
        ones_4 = np.array([1, 1, 5, 1, 1, 5, 5, 5, 5]).reshape([3, 3])
        nbeibour_1 = signal.correlate2d(edge_cp, ones_1, boundary='fill', mode='same')
        nbeibour_2 = signal.correlate2d(edge_cp, ones_2, boundary='fill', mode='same')
        nbeibour_3 = signal.correlate2d(edge_cp, ones_3, boundary='fill', mode='same')
        nbeibour_4 = signal.correlate2d(edge_cp, ones_4, boundary='fill', mode='same')
        for x in range(edge_cp.shape[0]):
            for y in range(edge_cp.shape[1]):
                if nbeibour_1[x, y] == 4 or nbeibour_2[x, y] == 4 or nbeibour_3[x, y] == 4 or nbeibour_4[x, y] == 4:
                    edge[x, y] = muhu_value
                    nbeibour_1[x - 1, y - 1] = nbeibour_1[x - 1, y - 1] - 1
                    nbeibour_1[x - 1, y] = nbeibour_1[x - 1, y] - 1
                    nbeibour_1[x, y - 1] = nbeibour_1[x, y - 1] - 1
                    nbeibour_1[x, y] = nbeibour_1[x, y] - 1

                    nbeibour_2[x - 1, y] = nbeibour_2[x - 1, y] - 1
                    nbeibour_2[x - 1, y + 1] = nbeibour_2[x - 1, y + 1] - 1
                    nbeibour_2[x, y] = nbeibour_2[x, y] - 1
                    nbeibour_2[x, y + 1] = nbeibour_2[x, y + 1] - 1

                    nbeibour_3[x + 1, y] = nbeibour_3[x + 1, y] - 1
                    nbeibour_3[x + 1, y - 1] = nbeibour_3[x + 1, y - 1] - 1
                    nbeibour_3[x, y] = nbeibour_3[x, y] - 1
                    nbeibour_3[x, y - 1] = nbeibour_3[x, y - 1] - 1

                    nbeibour_4[x + 1, y] = nbeibour_4[x + 1, y] - 1
                    nbeibour_4[x + 1, y + 1] = nbeibour_4[x + 1, y + 1] - 1
                    nbeibour_4[x, y] = nbeibour_4[x, y] - 1
                    nbeibour_4[x, y + 1] = nbeibour_4[x, y + 1] - 1
        return edge

    def dele_bound(edge):
        def getGrayDiff(img, currentPoint, tmpPoint):
            return img[currentPoint.x, currentPoint.y] * img[tmpPoint.x, tmpPoint.y]

        class Point(object):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def getX(self):
                return self.x

            def getY(self):
                return self.y

        def selectConnects(p):
            if p == 2:
                connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                            Point(-1, 1), Point(-1, 0), Point(-2, 0), Point(2, 0), Point(0, -2), Point(0, 2)]
            elif p == 3:
                connects = [Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)]
            elif p == 0:
                connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                            Point(-1, 1), Point(-1, 0)]
            else:
                connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
            return connects

        def regionGrow(img, nbeibour_1, seed, label, p=1):
            height, weight = img.shape
            seedMark = np.zeros(img.shape)
            seedList = []
            seedList.append(seed)
            connects = selectConnects(p)
            connects_edge = selectConnects(3)
            while (len(seedList) > 0):
                currentPoint = seedList.pop(0)
                seedMark[currentPoint.x, currentPoint.y] = label
                num = 4
                p_num = False
                for i in range(num):
                    tmpX = currentPoint.x + connects[i].x
                    tmpY = currentPoint.y + connects[i].y
                    if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                        continue
                    grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                    if grayDiff != 0 and seedMark[tmpX, tmpY] == 0:
                        p_num = True
                        if nbeibour_1[tmpX, tmpY] < 3:
                            p_num = True
                            seedMark[tmpX, tmpY] = label
                            seedList.append(Point(tmpX, tmpY))
                if not p_num:
                    tp_x = 0
                    tp_y = 0
                    numm = 0
                    for i in range(4):
                        tmpX = currentPoint.x + connects_edge[i].x
                        tmpY = currentPoint.y + connects_edge[i].y
                        if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                            continue
                        grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                        if grayDiff != 0 and seedMark[tmpX, tmpY] == 0:
                            if nbeibour_1[tmpX, tmpY] < 3:
                                tp_x = tmpX
                                tp_y = tmpY
                                numm = numm + 1
                    if numm == 1:
                        seedMark[tp_x, tp_y] = label
                        seedList.append(Point(tp_x, tp_y))

            return seedMark

        iso_point_map = iso_point(edge.copy())
        iso_point_grow = iso_point_map.copy()
        iso_point_grow[np.where(iso_point_grow == -1)] = 1

        ones_1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]).reshape([3, 3])
        nbeibour_1 = signal.correlate2d(iso_point_grow, ones_1, boundary='fill', mode='same')
        ones_2 = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape([3, 3])
        nbeibour_2 = signal.correlate2d(iso_point_grow, ones_2, boundary='fill', mode='same')
        ones_3 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0]).reshape([3, 3])
        nbeibour_3 = signal.correlate2d(iso_point_grow, ones_3, boundary='fill', mode='same')
        ones_4 = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape([3, 3])
        nbeibour_4 = signal.correlate2d(iso_point_grow, ones_4, boundary='fill', mode='same')
        ones_5 = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1]).reshape([3, 3])
        nbeibour_5 = signal.correlate2d(iso_point_grow, ones_5, boundary='fill', mode='same')
        nbeibour_1[np.where(nbeibour_2 == 3)] = 3
        nbeibour_1[np.where(nbeibour_3 == 3)] = 3
        nbeibour_1[np.where(nbeibour_4 == 3)] = 3
        nbeibour_1[np.where(nbeibour_5 == 3)] = 3

        edge_num = np.where(edge == 1)[0].shape[0]
        max_dele_num = min(int(edge_num * edge_dele_ratio), edge_dele_num)

        s1, s2 = np.where(iso_point_map == 1)
        label = 5
        for i in range(s1.shape[0]):
            x = s1[i]
            y = s2[i]
            seedmark = regionGrow(iso_point_grow, nbeibour_1, Point(x, y), label=label)
            if np.where(seedmark == label)[0].shape[0] < max_dele_num:
                edge[np.where(seedmark == label)] = muhu_value
            label = label + 1
        return edge

    def tune_bound(edge, dis_max_fine):
        max_op = dis_max_fine

        def getGrayDiff(img, currentPoint, tmpPoint):
            return img[currentPoint.x, currentPoint.y] * img[tmpPoint.x, tmpPoint.y]

        class Point(object):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def getX(self):
                return self.x

            def getY(self):
                return self.y

        def selectConnects(p):
            if p == 2:
                connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                            Point(-1, 1), Point(-1, 0), Point(-2, 0), Point(2, 0), Point(0, -2), Point(0, 2)]
            elif p == 3:
                connects = [Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)]
            elif p == 0:
                connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                            Point(-1, 1), Point(-1, 0)]
            else:
                connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
            return connects

        def regionGrow(img, nbeibour_1, seed, label, p=1):
            height, weight = img.shape
            seedMark = np.zeros(img.shape)
            seedList = []
            seedList.append(seed)
            connects = selectConnects(p)
            connects_edge = selectConnects(3)
            while (len(seedList) > 0):
                currentPoint = seedList.pop(0)
                seedMark[currentPoint.x, currentPoint.y] = label
                num = 4
                p_num = False
                for i in range(num):
                    tmpX = currentPoint.x + connects[i].x
                    tmpY = currentPoint.y + connects[i].y
                    if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                        continue
                    grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                    if grayDiff != 0 and seedMark[tmpX, tmpY] == 0:
                        p_num = True
                        if nbeibour_1[tmpX, tmpY] < 3:
                            seedMark[tmpX, tmpY] = label
                            seedList.append(Point(tmpX, tmpY))
                if not p_num:
                    tp_x = 0
                    tp_y = 0
                    numm = 0
                    for i in range(4):
                        tmpX = currentPoint.x + connects_edge[i].x
                        tmpY = currentPoint.y + connects_edge[i].y
                        if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                            continue
                        grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                        if grayDiff != 0 and seedMark[tmpX, tmpY] == 0:
                            if nbeibour_1[tmpX, tmpY] < 3:
                                tp_x = tmpX
                                tp_y = tmpY
                                numm = numm + 1
                    if numm == 1:
                        seedMark[tp_x, tp_y] = label
                        seedList.append(Point(tp_x, tp_y))

            return seedMark

        distance_x = edge.shape[0]
        distance_y = edge.shape[1]
        edge_pad = -1 * np.ones([edge.shape[0] + distance_x * 2, edge.shape[1] + distance_y * 2])
        edge_pad[distance_x:distance_x + edge.shape[0], distance_y:distance_y + edge.shape[1]] = edge
        edge = edge_pad
        edge_cp = edge.copy()

        iso_point_map = iso_point(edge.copy())  # 只有孤立点是1 其他的边界是-1 非边界是0

        iso_point_grow = iso_point_map.copy()  # 边界是1 非边界是0
        iso_point_grow[np.where(iso_point_grow == -1)] = 1

        ones_1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]).reshape([3, 3])
        nbeibour_1 = signal.correlate2d(iso_point_grow, ones_1, boundary='fill', mode='same')
        ones_2 = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape([3, 3])
        nbeibour_2 = signal.correlate2d(iso_point_grow, ones_2, boundary='fill', mode='same')
        ones_3 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0]).reshape([3, 3])
        nbeibour_3 = signal.correlate2d(iso_point_grow, ones_3, boundary='fill', mode='same')
        ones_4 = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]).reshape([3, 3])
        nbeibour_4 = signal.correlate2d(iso_point_grow, ones_4, boundary='fill', mode='same')
        ones_5 = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1]).reshape([3, 3])
        nbeibour_5 = signal.correlate2d(iso_point_grow, ones_5, boundary='fill', mode='same')
        nbeibour_1[np.where(nbeibour_2 == 3)] = 3
        nbeibour_1[np.where(nbeibour_3 == 3)] = 3
        nbeibour_1[np.where(nbeibour_4 == 3)] = 3
        nbeibour_1[np.where(nbeibour_5 == 3)] = 3  # 有一些3表示间隔点

        edge_shuxing = iso_point_shuxing(iso_point_map, edge.copy(), nbeibour_1, raugh=True)

        edges_pad_cp = edge.copy()
        s1, s2 = np.where(iso_point_map == 1)
        label = 5
        con_value = 100
        pp_con_value = 105
        for i in range(s1.shape[0]):
            x = s1[i]
            y = s2[i]

            if x == 501:
                print('ss')
            if iso_point_map[x, y] == -1:
                continue
            seedmark = regionGrow(iso_point_grow, nbeibour_1, Point(x, y), label=label)
            if edge_shuxing[x, y] == -1:
                edge[np.where(seedmark == label)] = muhu_value
            r1, r2 = np.where(seedmark == label)
            dis_max = 0
            for r1_idx in range(len(r1)):
                dis_max = max(dis_max, int(np.sqrt((r1[r1_idx] - x) ** 2 + (r2[r1_idx] - y) ** 2)))
            # dis_max = max(max(max(r1) - x, x - min(r1)), max(max(r2) - y, y - min(r2)))
            flag = True
            if dis_max > max_op:  # 注定不会删除
                dis_max = max_op

            if edge_shuxing[x, y] == 0:
                edge[x, y] = 0

                e1, e2 = np.where(edge[x - dis_max:x + dis_max + 1, y - dis_max:y + dis_max + 1] == 1)
                if len(e1) > 0:
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, dis_max, dis_max)
                    if min_dis_opt <= dis_max:
                        x_ = e1[opt_idx] + x - dis_max
                        y_ = e2[opt_idx] + y - dis_max

                        x_need = np.abs(x - x_)
                        y_need = np.abs(y - y_)
                        if x_need > y_need:
                            if x > x_:
                                x_start = x_
                                y_start = y_
                                x_end = x
                                y_end = y
                            else:
                                x_start = x
                                y_start = y
                                x_end = x_
                                y_end = y_
                            it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                            for ii in range(1, x_need + 1):
                                edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                        else:
                            if y > y_:
                                x_start = x_
                                y_start = y_
                                x_end = x
                                y_end = y
                            else:
                                x_start = x
                                y_start = y
                                x_end = x_
                                y_end = y_
                            it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                            for ii in range(1, y_need + 1):
                                edge[x_start + int(ii * it_di), y_start + ii] = con_value
                        flag = False
                edge[x, y] = 1

            elif edge_shuxing[x, y] == 1:
                edge[x, y] = 0
                iso_point_map[x, y] = -1
                e1, e2 = np.where(iso_point_map[x:x + max_op + 1, y - max_op:y + max_op + 1] == 1)
                if len(e1) > 0:
                    num_pp_con = 0
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, 0, max_op)
                    while (num_pp_con < len(e1)):
                        num_pp_con += 1
                        if min_dis_opt <= max_op:
                            x_ = e1[opt_idx] + x - 0
                            y_ = e2[opt_idx] + y - max_op
                            if (edge_shuxing[x_, y_] == 1 and x_ <= x) or (edge_shuxing[x_, y_] == 2 and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 3 and x_ >= x) or (
                                    edge_shuxing[x_, y_] == 4 and y_ <= y) or \
                                    (edge_shuxing[x_, y_] == 5 and x_ <= x and y_ >= y) or (
                                    edge_shuxing[x_, y_] == 6 and x_ >= x and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 7 and x_ >= x and y_ <= y) or (
                                    edge_shuxing[x_, y_] == 8 and x_ <= x and y_ <= y):
                                cross_edge = False
                                edge_old = edge.copy()
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        if edge[x_start + ii, y_start + int(ii * it_di)] == 1:
                                            cross_edge = True
                                        edge[x_start + ii, y_start + int(ii * it_di)] = pp_con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        if edge[x_start + int(ii * it_di), y_start + ii] == 1:
                                            cross_edge = True
                                        edge[x_start + int(ii * it_di), y_start + ii] = pp_con_value

                                if cross_edge:
                                    edge = edge_old
                                else:
                                    edge[np.where(edge == pp_con_value)] = con_value
                                    flag = False
                                    break

                if flag == True:
                    e1, e2 = np.where(edge[x:x + dis_max + 1, y - dis_max:y + dis_max + 1] == 1)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, 0, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - 0
                                y_ = e2[dis_idx] + y - dis_max
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 1
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                if flag == True:
                    e1, e2 = np.where(edge[x:x + dis_max + 1, y - dis_max:y + dis_max + 1] == 2)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, 0, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - 0
                                y_ = e2[dis_idx] + y - dis_max
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 2
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                edge[x, y] = 1
                iso_point_map[x, y] = 1

            elif edge_shuxing[x, y] == 2:
                edge[x, y] = 0
                iso_point_map[x, y] = -1
                e1, e2 = np.where(edge[x - max_op:x + max_op + 1, y - max_op:y + 1] == 1)
                if len(e1) > 0:
                    num_pp_con = 0
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, max_op, max_op)
                    while (num_pp_con < len(e1)):
                        num_pp_con += 1
                        if min_dis_opt <= max_op:
                            x_ = e1[opt_idx] + x - max_op
                            y_ = e2[opt_idx] + y - max_op
                            if (edge_shuxing[x_, y_] == 1 and x_ <= x) or (edge_shuxing[x_, y_] == 2 and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 3 and x_ >= x) or (
                                    edge_shuxing[x_, y_] == 4 and y_ <= y) or \
                                    (edge_shuxing[x_, y_] == 5 and x_ <= x and y_ >= y) or (
                                    edge_shuxing[x_, y_] == 6 and x_ >= x and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 7 and x_ >= x and y_ <= y) or (
                                    edge_shuxing[x_, y_] == 8 and x_ <= x and y_ <= y):
                                cross_edge = False
                                edge_old = edge.copy()
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        if edge[x_start + ii, y_start + int(ii * it_di)] == 1:
                                            cross_edge = True
                                        edge[x_start + ii, y_start + int(ii * it_di)] = pp_con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        if edge[x_start + int(ii * it_di), y_start + ii] == 1:
                                            cross_edge = True
                                        edge[x_start + int(ii * it_di), y_start + ii] = pp_con_value

                                if cross_edge:
                                    edge = edge_old
                                else:
                                    edge[np.where(edge == pp_con_value)] = con_value
                                    flag = False
                                    break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + dis_max + 1, y - dis_max:y + 1] == 1)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - dis_max

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 1
                                neg_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + dis_max + 1, y - dis_max:y + 1] == 2)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - dis_max

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 2
                                neg_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                edge[x, y] = 1
                iso_point_map[x, y] = 1

            elif edge_shuxing[x, y] == 3:
                edge[x, y] = 0
                iso_point_map[x, y] = -1
                e1, e2 = np.where(iso_point_map[x - max_op:x + 1, y - max_op:y + max_op + 1] == 1)
                if len(e1) > 0:
                    num_pp_con = 0
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, max_op, max_op)
                    while (num_pp_con < len(e1)):
                        num_pp_con += 1
                        if min_dis_opt <= max_op:
                            x_ = e1[opt_idx] + x - max_op
                            y_ = e2[opt_idx] + y - max_op
                            if (edge_shuxing[x_, y_] == 1 and x_ <= x) or (edge_shuxing[x_, y_] == 2 and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 3 and x_ >= x) or (
                                    edge_shuxing[x_, y_] == 4 and y_ <= y) or \
                                    (edge_shuxing[x_, y_] == 5 and x_ <= x and y_ >= y) or (
                                    edge_shuxing[x_, y_] == 6 and x_ >= x and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 7 and x_ >= x and y_ <= y) or (
                                    edge_shuxing[x_, y_] == 8 and x_ <= x and y_ <= y):
                                cross_edge = False
                                edge_old = edge.copy()
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        if edge[x_start + ii, y_start + int(ii * it_di)] == 1:
                                            cross_edge = True
                                        edge[x_start + ii, y_start + int(ii * it_di)] = pp_con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        if edge[x_start + int(ii * it_di), y_start + ii] == 1:
                                            cross_edge = True
                                        edge[x_start + int(ii * it_di), y_start + ii] = pp_con_value

                                if cross_edge:
                                    edge = edge_old
                                else:
                                    edge[np.where(edge == pp_con_value)] = con_value
                                    flag = False
                                    break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + 1, y - dis_max:y + dis_max + 1] == 1)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - dis_max

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 1
                                neg_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[
                                        np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + 1, y - dis_max:y + dis_max + 1] == 2)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - dis_max

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 2
                                neg_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[
                                        np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                edge[x, y] = 1
                iso_point_map[x, y] = 1

            elif edge_shuxing[x, y] == 4:
                edge[x, y] = 0
                iso_point_map[x, y] = -1
                e1, e2 = np.where(iso_point_map[x - max_op:x + max_op + 1, y:y + max_op + 1] == 1)
                if len(e1) > 0:
                    num_pp_con = 0
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, max_op, 0)
                    while (num_pp_con < len(e1)):
                        num_pp_con += 1
                        if min_dis_opt <= max_op:
                            x_ = e1[opt_idx] + x - max_op
                            y_ = e2[opt_idx] + y - 0
                            if (edge_shuxing[x_, y_] == 1 and x_ <= x) or (edge_shuxing[x_, y_] == 2 and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 3 and x_ >= x) or (
                                    edge_shuxing[x_, y_] == 4 and y_ <= y) or \
                                    (edge_shuxing[x_, y_] == 5 and x_ <= x and y_ >= y) or (
                                    edge_shuxing[x_, y_] == 6 and x_ >= x and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 7 and x_ >= x and y_ <= y) or (
                                    edge_shuxing[x_, y_] == 8 and x_ <= x and y_ <= y):
                                cross_edge = False
                                edge_old = edge.copy()
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        if edge[x_start + ii, y_start + int(ii * it_di)] == 1:
                                            cross_edge = True
                                        edge[x_start + ii, y_start + int(ii * it_di)] = pp_con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        if edge[x_start + int(ii * it_di), y_start + ii] == 1:
                                            cross_edge = True
                                        edge[x_start + int(ii * it_di), y_start + ii] = pp_con_value

                                if cross_edge:
                                    edge = edge_old
                                else:
                                    edge[np.where(edge == pp_con_value)] = con_value
                                    flag = False
                                    break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + dis_max + 1, y:y + dis_max + 1] == 1)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, 0)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - 0

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 1
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[
                                        np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + dis_max + 1, y:y + dis_max + 1] == 2)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, 0)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - 0

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 2
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[
                                        np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break
                edge[x, y] = 1
                iso_point_map[x, y] = 1

            elif edge_shuxing[x, y] == 5:
                edge[x, y] = 0
                iso_point_map[x, y] = -1
                e1, e2 = np.where(iso_point_map[x:x + max_op + 1, y - max_op:y + 1] == 1)
                if len(e1) > 0:
                    num_pp_con = 0
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, 0, max_op)
                    while (num_pp_con < len(e1)):
                        num_pp_con += 1
                        if min_dis_opt <= max_op:
                            x_ = e1[opt_idx] + x - 0
                            y_ = e2[opt_idx] + y - max_op
                            if (edge_shuxing[x_, y_] == 1 and x_ <= x) or (edge_shuxing[x_, y_] == 2 and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 3 and x_ >= x) or (
                                    edge_shuxing[x_, y_] == 4 and y_ <= y) or \
                                    (edge_shuxing[x_, y_] == 5 and x_ <= x and y_ >= y) or (
                                    edge_shuxing[x_, y_] == 6 and x_ >= x and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 7 and x_ >= x and y_ <= y) or (
                                    edge_shuxing[x_, y_] == 8 and x_ <= x and y_ <= y):
                                cross_edge = False
                                edge_old = edge.copy()
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        if edge[x_start + ii, y_start + int(ii * it_di)] == 1:
                                            cross_edge = True
                                        edge[x_start + ii, y_start + int(ii * it_di)] = pp_con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        if edge[x_start + int(ii * it_di), y_start + ii] == 1:
                                            cross_edge = True
                                        edge[x_start + int(ii * it_di), y_start + ii] = pp_con_value

                                if cross_edge:
                                    edge = edge_old
                                else:
                                    edge[np.where(edge == pp_con_value)] = con_value
                                    flag = False
                                    break

                if flag == True:
                    e1, e2 = np.where(edge[x:x + dis_max + 1, y - dis_max:y + 1] == 1)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, 0, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - 0
                                y_ = e2[dis_idx] + y - dis_max

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 1
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                if flag == True:
                    e1, e2 = np.where(edge[x:x + dis_max + 1, y - dis_max:y + 1] == 2)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, 0, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - 0
                                y_ = e2[dis_idx] + y - dis_max

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 2
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break
                edge[x, y] = 1
                iso_point_map[x, y] = 1

            elif edge_shuxing[x, y] == 6:
                edge[x, y] = 0
                iso_point_map[x, y] = -1
                e1, e2 = np.where(iso_point_map[x - max_op:x + 1, y - max_op:y + 1] == 1)
                if len(e1) > 0:
                    num_pp_con = 0
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, max_op, max_op)
                    while (num_pp_con < len(e1)):
                        num_pp_con += 1
                        if min_dis_opt <= max_op:
                            x_ = e1[opt_idx] + x - max_op
                            y_ = e2[opt_idx] + y - max_op
                            if (edge_shuxing[x_, y_] == 1 and x_ <= x) or (edge_shuxing[x_, y_] == 2 and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 3 and x_ >= x) or (
                                    edge_shuxing[x_, y_] == 4 and y_ <= y) or \
                                    (edge_shuxing[x_, y_] == 5 and x_ <= x and y_ >= y) or (
                                    edge_shuxing[x_, y_] == 6 and x_ >= x and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 7 and x_ >= x and y_ <= y) or (
                                    edge_shuxing[x_, y_] == 8 and x_ <= x and y_ <= y):
                                cross_edge = False
                                edge_old = edge.copy()
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        if edge[x_start + ii, y_start + int(ii * it_di)] == 1:
                                            cross_edge = True
                                        edge[x_start + ii, y_start + int(ii * it_di)] = pp_con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        if edge[x_start + int(ii * it_di), y_start + ii] == 1:
                                            cross_edge = True
                                        edge[x_start + int(ii * it_di), y_start + ii] = pp_con_value

                                if cross_edge:
                                    edge = edge_old
                                else:
                                    edge[np.where(edge == pp_con_value)] = con_value
                                    flag = False
                                    break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + 1, y - dis_max:y + 1] == 1)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - dis_max

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 1
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[
                                        np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + 1, y - dis_max:y + 1] == 2)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, dis_max)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - dis_max

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 2
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[
                                        np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break
                edge[x, y] = 1
                iso_point_map[x, y] = 1

            elif edge_shuxing[x, y] == 7:
                edge[x, y] = 0
                iso_point_map[x, y] = -1
                e1, e2 = np.where(iso_point_map[x - max_op:x + 1, y:y + max_op + 1] == 1)
                if len(e1) > 0:
                    num_pp_con = 0
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, max_op, 0)
                    while (num_pp_con < len(e1)):
                        num_pp_con += 1
                        if min_dis_opt <= max_op:
                            x_ = e1[opt_idx] + x - max_op
                            y_ = e2[opt_idx] + y - 0
                            if (edge_shuxing[x_, y_] == 1 and x_ <= x) or (edge_shuxing[x_, y_] == 2 and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 3 and x_ >= x) or (
                                    edge_shuxing[x_, y_] == 4 and y_ <= y) or \
                                    (edge_shuxing[x_, y_] == 5 and x_ <= x and y_ >= y) or (
                                    edge_shuxing[x_, y_] == 6 and x_ >= x and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 7 and x_ >= x and y_ <= y) or (
                                    edge_shuxing[x_, y_] == 8 and x_ <= x and y_ <= y):
                                cross_edge = False
                                edge_old = edge.copy()
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        if edge[x_start + ii, y_start + int(ii * it_di)] == 1:
                                            cross_edge = True
                                        edge[x_start + ii, y_start + int(ii * it_di)] = pp_con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        if edge[x_start + int(ii * it_di), y_start + ii] == 1:
                                            cross_edge = True
                                        edge[x_start + int(ii * it_di), y_start + ii] = pp_con_value

                                if cross_edge:
                                    edge = edge_old
                                else:
                                    edge[np.where(edge == pp_con_value)] = con_value
                                    flag = False
                                    break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + 1, y:y + dis_max + 1] == 1)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, 0)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - 0

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 1
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[
                                        np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                if flag == True:
                    e1, e2 = np.where(edge[x - dis_max:x + 1, y:y + dis_max + 1] == 2)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, dis_max, 0)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - dis_max
                                y_ = e2[dis_idx] + y - 0

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                edge[x_, y_] = 2
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                    np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[
                                        np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break
                edge[x, y] = 1
                iso_point_map[x, y] = 1

            elif edge_shuxing[x, y] == 8:
                edge[x, y] = 0
                iso_point_map[x, y] = -1
                e1, e2 = np.where(iso_point_map[x:x + max_op + 1, y:y + max_op + 1] == 1)
                if len(e1) > 0:
                    num_pp_con = 0
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, 0, 0)
                    while (num_pp_con < len(e1)):
                        num_pp_con += 1
                        if min_dis_opt <= max_op:
                            x_ = e1[opt_idx] + x - 0
                            y_ = e2[opt_idx] + y - 0
                            if (edge_shuxing[x_, y_] == 1 and x_ <= x) or (edge_shuxing[x_, y_] == 2 and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 3 and x_ >= x) or (
                                    edge_shuxing[x_, y_] == 4 and y_ <= y) or \
                                    (edge_shuxing[x_, y_] == 5 and x_ <= x and y_ >= y) or (
                                    edge_shuxing[x_, y_] == 6 and x_ >= x and y_ >= y) or \
                                    (edge_shuxing[x_, y_] == 7 and x_ >= x and y_ <= y) or (
                                    edge_shuxing[x_, y_] == 8 and x_ <= x and y_ <= y):
                                cross_edge = False
                                edge_old = edge.copy()
                                edge[x_, y_] = 0
                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        if edge[x_start + ii, y_start + int(ii * it_di)] == 1:
                                            cross_edge = True
                                        edge[x_start + ii, y_start + int(ii * it_di)] = pp_con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        if edge[x_start + int(ii * it_di), y_start + ii] == 1:
                                            cross_edge = True
                                        edge[x_start + int(ii * it_di), y_start + ii] = pp_con_value

                                if cross_edge:
                                    edge = edge_old
                                else:
                                    edge[np.where(edge == pp_con_value)] = con_value
                                    flag = False
                                    break

                if flag == True:
                    e1, e2 = np.where(edge[x:x + dis_max + 1, y:y + dis_max + 1] == 1)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, 0, 0)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - 0
                                y_ = e2[dis_idx] + y - 0

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                edge[x_, y_] = 1
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break

                if flag == True:
                    e1, e2 = np.where(edge[x:x + dis_max + 1, y:y + dis_max + 1] == 2)
                    if len(e1) > 0:
                        dis_jihe = min_dis_getjihe(e1, e2, 0, 0)
                        dis_idx_jihe = np.argsort(dis_jihe)
                        for dis_idx in dis_idx_jihe:
                            if dis_jihe[dis_idx] <= dis_max:
                                x_ = e1[dis_idx] + x - 0
                                y_ = e2[dis_idx] + y - 0

                                x_need = np.abs(x - x_)
                                y_need = np.abs(y - y_)
                                if x_need > y_need:
                                    if x > x_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                                    for ii in range(1, x_need + 1):
                                        edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                                else:
                                    if y > y_:
                                        x_start = x_
                                        y_start = y_
                                        x_end = x
                                        y_end = y
                                    else:
                                        x_start = x
                                        y_start = y
                                        x_end = x_
                                        y_end = y_
                                    it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                                    for ii in range(1, y_need + 1):
                                        edge[x_start + int(ii * it_di), y_start + ii] = con_value
                                neg_num_connect = np.where(edges_pad_cp[np.where(edge == con_value)] == 0)[0].shape[0]
                                fuz_num_connect = \
                                np.where(edges_pad_cp[np.where(edge == con_value)] == muhu_value)[0].shape[0]
                                edge[x_, y_] = 2
                                if neg_num_connect > (neg_num_connect + fuz_num_connect) * conne_ratio:
                                    edge[np.where(edge == con_value)] = edges_pad_cp[np.where(edge == con_value)]
                                else:
                                    edge[np.where(edge == con_value)] = 1
                                    flag = False
                                    break
                                edge[x, y] = 0
                            else:
                                break
                edge[x, y] = 1
                iso_point_map[x, y] = 1

            p1 = edge_cp[np.where(edge == con_value)]

            if flag:
                edge[np.where(seedmark == label)] = 2
                edge[x, y] = 0
                dis_max_undirect = int(dis_max / 2)
                e1, e2 = np.where(edge[x - dis_max_undirect:x + dis_max_undirect + 1,
                                  y - dis_max_undirect:y + dis_max_undirect + 1] == 1)
                if len(e1) > 0:
                    opt_idx, min_dis_opt = min_dis_get(e1, e2, dis_max_undirect, dis_max_undirect)
                    if min_dis_opt <= dis_max_undirect:
                        x_ = e1[opt_idx] + x - dis_max_undirect
                        y_ = e2[opt_idx] + y - dis_max_undirect

                        x_need = np.abs(x - x_)
                        y_need = np.abs(y - y_)
                        if x_need > y_need:
                            if x > x_:
                                x_start = x_
                                y_start = y_
                                x_end = x
                                y_end = y
                            else:
                                x_start = x
                                y_start = y
                                x_end = x_
                                y_end = y_
                            it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                            for ii in range(1, x_need + 1):
                                edge[x_start + ii, y_start + int(ii * it_di)] = con_value
                        else:
                            if y > y_:
                                x_start = x_
                                y_start = y_
                                x_end = x
                                y_end = y
                            else:
                                x_start = x
                                y_start = y
                                x_end = x_
                                y_end = y_
                            it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                            for ii in range(1, y_need + 1):
                                edge[x_start + int(ii * it_di), y_start + ii] = con_value
                        flag = False
                edge[x, y] = 1

                edge[np.where(seedmark == label)] = 1

            if flag and dis_max < max_op:
                edge[np.where(seedmark == label)] = muhu_value
                iso_point_map[np.where(seedmark == label)] = -1
            else:
                ones_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape([3, 3])
                nbeibour_ = signal.correlate2d(edge, ones_1, boundary='fill', mode='same')
                iso_point_map[np.logical_and(nbeibour_ >= 100, iso_point_map == 1)] = -1  # 附近的点不再是孤立点了
                iso_point_grow[np.logical_and(nbeibour_ >= 100, iso_point_grow == 1)] = iso_point_grow[np.logical_and(
                    nbeibour_ >= 100, iso_point_grow == 1)] + 1  # 附近的点不再是孤立点隔离
                edge[np.where(edge == con_value)] = 1
            label = label + 1

        return edge[distance_x:distance_x + distance_x, distance_y:distance_y + distance_y]

    def hard_connection(edges, distance=6):
        def getGrayDiff(img, currentPoint, tmpPoint):
            return np.abs(img[currentPoint.x, currentPoint.y] - img[tmpPoint.x, tmpPoint.y])

        class Point(object):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def getX(self):
                return self.x

            def getY(self):
                return self.y

        def selectConnects(p):
            if p == 2:
                connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                            Point(-1, 1),
                            Point(-1, 0), Point(-2, 0), Point(2, 0), Point(0, -2), Point(0, 2)]
            elif p == 0:
                connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1),
                            Point(-1, 1),
                            Point(-1, 0)]
            else:
                connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
            return connects

        def regionGrow(img, seed, label, p=0):
            height, weight = img.shape
            seedMark = np.zeros(img.shape)
            seedList = []
            seedList.append(seed)
            connects = selectConnects(p)
            while (len(seedList) > 0):
                currentPoint = seedList.pop(0)
                seedMark[currentPoint.x, currentPoint.y] = label
                num = 8
                for i in range(num):
                    tmpX = currentPoint.x + connects[i].x
                    tmpY = currentPoint.y + connects[i].y
                    if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                        continue
                    grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                    if grayDiff == 0 and seedMark[tmpX, tmpY] == 0:
                        seedMark[tmpX, tmpY] = label
                        seedList.append(Point(tmpX, tmpY))
            return seedMark

        iso_point_map = iso_point(edges.copy())
        edge_shuxing_pad = -1 * np.ones([iso_point_map.shape[0] + distance * 2, iso_point_map.shape[1] + distance * 2])
        edges_pad = 0 * np.ones([edges.shape[0] + distance * 2, edges.shape[1] + distance * 2])
        edge_shuxing_pad[distance:distance + iso_point_map.shape[0],
        distance:distance + iso_point_map.shape[1]] = iso_point_map
        edges_pad[distance:distance + edges.shape[0], distance:distance + edges.shape[1]] = edges
        s1, s2 = np.where(edge_shuxing_pad == 1)
        for i in range(s1.shape[0]):
            x = s1[i]
            y = s2[i]
            edge_tmp = edges_pad[x - distance:x + distance + 1, y - distance:y + distance + 1]
            seedmark = regionGrow(edge_tmp, Point(distance, distance), label=10)
            q1, q2 = np.where(seedmark == 10)

            edges_pad[q1 + x - distance, q2 + y - distance] = 10

            edges_pad[x, y] = 0
            e1, e2 = np.where(edges_pad[x - distance:x + distance + 1, y - distance:y + distance + 1] == 1)
            if len(e1) > 0:
                dis_jihe = min_dis_getjihe(e1, e2, distance, distance)
                dis_idx_jihe = np.argsort(dis_jihe)
                for dis_idx in dis_idx_jihe:
                    if dis_jihe[dis_idx] <= distance:
                        x_ = e1[dis_idx] + x - distance
                        y_ = e2[dis_idx] + y - distance
                        x_need = np.abs(x - x_)
                        y_need = np.abs(y - y_)
                        if x_need > y_need:
                            if x > x_:
                                x_start = x_
                                y_start = y_
                                x_end = x
                                y_end = y
                            else:
                                x_start = x
                                y_start = y
                                x_end = x_
                                y_end = y_
                            it_di = float(y_need) / float(x_need) * np.sign(y_end - y_start)
                            for ii in range(1, x_need + 1):
                                edges_pad[x_start + ii, y_start + int(ii * it_di)] = con_value
                        else:
                            if y > y_:
                                x_start = x_
                                y_start = y_
                                x_end = x
                                y_end = y
                            else:
                                x_start = x
                                y_start = y
                                x_end = x_
                                y_end = y_
                            it_di = float(x_need) / float(y_need) * np.sign(x_end - x_start)
                            for ii in range(1, y_need + 1):
                                edges_pad[x_start + int(ii * it_di), y_start + ii] = con_value
                        break
                    else:
                        break

            edges_pad[x, y] = 1
            edges_pad[q1 + x - distance, q2 + y - distance] = 1

        edges = edges_pad[distance:distance + edges.shape[0], distance:distance + edges.shape[1]]
        return edges

    def seed_mark_add(seed_mark, x, y, bound_lab):
        # 输入 边界实例图，位置，当前实例索引
        # 返回 边界实例图，当前索引，是否需要更新索引

        bound_st = set()
        flag = False

        if seed_mark[x - 1, y - 1] > 0:
            bound_st.add(seed_mark[x - 1, y - 1])
        if seed_mark[x - 1, y] > 0:
            bound_st.add(seed_mark[x - 1, y])
        if seed_mark[x - 1, y + 1] > 0:
            bound_st.add(seed_mark[x - 1, y + 1])
        if seed_mark[x, y - 1] > 0:
            bound_st.add(seed_mark[x, y - 1])
        if seed_mark[x, y + 1] > 0:
            bound_st.add(seed_mark[x, y + 1])
        if seed_mark[x + 1, y - 1] > 0:
            bound_st.add(seed_mark[x + 1, y - 1])
        if seed_mark[x + 1, y] > 0:
            bound_st.add(seed_mark[x + 1, y])
        if seed_mark[x + 1, y + 1] > 0:
            bound_st.add(seed_mark[x + 1, y + 1])
        if len(bound_st) == 0:
            seed_mark[x, y] = bound_lab
            flag = True

        elif len(bound_st) == 1:
            seed_mark[x, y] = bound_st.pop()
        else:
            lab_tmp = bound_st.pop()
            seed_mark[x, y] = lab_tmp
            while (len(bound_st) > 0):
                lab_new = bound_st.pop()
                seed_mark[np.where(seed_mark == lab_new)] = lab_tmp

        return seed_mark, seed_mark[x, y], flag

    epoch_globel, image_name, region_min, region_min_second, region_min_ratio, edge_dele_ratio, edge_dele_num, threhold_, \
    UNLAB_TRAIN_EDGE_Pseudo_FLIST, perued_num, min_distance, hard_distance, weight_neg_pos, conne_dis, \
    pos_neibour_max, tune_flag, negpos_ratio, conne_ratio, dis_max_fine, epoch_could_tune, threhold_fugai, epoch_could_pos_tune, data_mode = arg


    muhu_value = 0.501960813999176
    con_value = 1
    pad_num = 2

    flag_e = 'F'  # 负像素全给定
    flag_t = 'F'  # 闭合修正
    # 正负样本不同的阈值
    threhold = threhold_
    neg_threhold = threhold_

    positive = np.load(UNLAB_TRAIN_EDGE_Pseudo_FLIST + '/' + image_name.replace('.png', '_pos.npy'))
    negtive = np.load(UNLAB_TRAIN_EDGE_Pseudo_FLIST + '/' + image_name.replace('.png', '_neg.npy'))
    edges = cv2.imread(UNLAB_TRAIN_EDGE_Pseudo_FLIST + '/' + image_name, 0)
    neg_expand = cv2.imread(UNLAB_TRAIN_EDGE_Pseudo_FLIST + '/' + image_name.split('.')[0] + '_neg_expand.png', 0)

    if len(np.where(neg_expand==255)[0]) > 0 :
        conne_ratio = 1
        region_min_ratio = -1
        if data_mode == 'gray':
            edge_dele_num = int(edge_dele_num * 1.5)
            conne_dis = int(conne_dis * 1.5)

    edges = edges.astype(np.float)  # lmc for linux
    edges[np.where(edges == 128)] = muhu_value  # lmc for linux
    edges[np.where(edges == 255)] = 1  # lmc for linux
    edges[np.where(edges == 0)] = 0  # lmc for linux

    height = edges.shape[0]
    width = edges.shape[1]
    ratio_decent = 1

    negtive_pad = neg_threhold * np.ones([negtive.shape[0] + pad_num * 2, negtive.shape[1] + pad_num * 2])
    negtive_pad[pad_num:pad_num + negtive.shape[0], pad_num:pad_num + negtive.shape[1]] = negtive
    positive_pad = threhold * np.ones([positive.shape[0] + pad_num * 2, positive.shape[1] + pad_num * 2])
    positive_pad[pad_num:pad_num + positive.shape[0], pad_num:pad_num + positive.shape[1]] = positive
    edges_pad = -1 * np.ones([edges.shape[0] + pad_num * 2, edges.shape[1] + pad_num * 2])
    edges_pad[pad_num:pad_num + edges.shape[0], pad_num:pad_num + edges.shape[1]] = edges

    if data_mode == 'rgb':
        mask = cv2.imread(UNLAB_TRAIN_EDGE_Pseudo_FLIST + '/' + image_name.split('.')[0] + '_mask.png', 0)
        mask_pad = -1 * np.ones([edges.shape[0] + pad_num * 2, edges.shape[1] + pad_num * 2])
        mask_pad[pad_num:pad_num + edges.shape[0], pad_num:pad_num + edges.shape[1]] = mask
        mask = mask_pad

    neg_expand_pad = -1 * np.ones([neg_expand.shape[0] + pad_num * 2, neg_expand.shape[1] + pad_num * 2])
    neg_expand_pad[pad_num:pad_num + neg_expand.shape[0], pad_num:pad_num + neg_expand.shape[1]] = neg_expand

    edges = edges_pad
    positive = positive_pad
    negtive = negtive_pad
    weight_matrix = (positive - negtive).copy()
    # negtive_pad_cp = negtive_pad.copy()
    threshold_p = threhold
    threshold_n = neg_threhold


    num = int(perued_num * (1 / (1 - negpos_ratio)) / weight_neg_pos)
    num_cp = num
    num_totla_neg = num

    # 做更新 将一刀切的恢复
    edges[np.logical_and(neg_expand_pad == 255, edges == 0)] = muhu_value
    negtive[np.where(edges == 0)] = neg_threhold

    while (num > 0):
        x_a, y_b = np.unravel_index(np.argmax(negtive), negtive.shape)
        if negtive[x_a, y_b] <= threshold_n:
            break
        if x_a < pad_num or x_a >= pad_num + height or y_b < pad_num or y_b >= pad_num + width:
            negtive[x_a, y_b] = threshold_n
            continue
        if edges[x_a, y_b] != 0:
            if edges[x_a, y_b] != 1:
                edges[x_a, y_b] = 0
                negtive[x_a, y_b] = threshold_n
                num = num - 1
            elif edges[x_a, y_b] == 1 and negtive[x_a, y_b] >= threhold_fugai:
                edges[x_a, y_b] = 0
                negtive[x_a, y_b] = threshold_n
                num = num - 1
            else:
                negtive[x_a, y_b] = threshold_n
        else:
            negtive[x_a, y_b] = threshold_n
    num_totla_neg = num_totla_neg - num

    # 做更新 将一刀切的恢复
    neg_expand_pad[np.logical_and(neg_expand_pad == 255, edges == 0)] = 0
    edges[np.logical_and(neg_expand_pad == 255, edges == muhu_value)] = 0


    if data_mode == 'rgb':
        nbeibour, positive_advance, shuxing_advance = get_advance_mask(edges.copy(), positive, mask)
    else:
        nbeibour, positive_advance, shuxing_advance = get_advance(edges.copy(), positive)
    positive[np.where(edges == 1)] = threhold



    bound_lab = 1
    s1, s2 = np.where(edges == 1)
    seedmark_record = -1 * np.ones(edges.shape)  # 边界各异的标识
    for i in range(s1.shape[0]):
        x = s1[i]
        y = s2[i]
        if seedmark_record[x, y] == -1 and edges[x, y] == 1:
            seed = Point(x, y)
            seedmark_s = regionGrow(edges, seed, label=bound_lab)
            seedmark_record[seedmark_s == bound_lab] = bound_lab
            bound_lab = bound_lab + 1

    num = 0


    pos_num_raw = np.where(edges == 1)[0].shape[0]
    neg_num_raw = np.where(edges == 0)[0].shape[0]

    pos_need_num = perued_num
    if data_mode == 'rgb':
        while (num < pos_need_num):
            x_a, y_b = np.unravel_index(np.argmax(positive_advance), positive_advance.shape)
            if positive_advance[x_a, y_b] <= threshold_p:
                nbeibour_7 = pos_neibour(edges.copy(), kernal=int(min_distance))
                positive_cp = positive.copy()
                positive_cp[np.where(edges == 1)] = threshold_p
                positive_cp[np.where(nbeibour_7 > 0)] = threshold_p
                if data_mode == 'rgb':
                    positive_cp[np.where(mask == 255)] = threshold_p
                x_a, y_b = np.unravel_index(np.argmax(positive_cp), positive_cp.shape)
                if positive_cp[x_a, y_b] <= threshold_p:
                    break
                positive[x_a, y_b] = threshold_p
                shuxing_advance[x_a, y_b] = 0

            if x_a < pad_num or x_a >= pad_num + height or y_b < pad_num or y_b >= pad_num + width:
                positive_advance[x_a, y_b] = threshold_p
                continue
            if not valit(nbeibour, x_a, y_b):
                positive_advance[x_a, y_b] = threshold_p
                continue
            if edges[x_a, y_b] != 1:
                edges[x_a, y_b] = 1
                seedmark_record, bound_lab_tmp, need_update = seed_mark_add(seedmark_record, x_a, y_b, bound_lab)
                if need_update:
                    bound_lab = bound_lab + 1
                positive_advance[x_a, y_b] = threshold_p
                nbeibour = pos_neibour_add(nbeibour, x_a, y_b)
                num = num + 1

                if x_a > pad_num + hard_distance and y_b > hard_distance + pad_num and x_a < pad_num + height - hard_distance and y_b < pad_num + width - hard_distance:
                    flag_autocon = False
                    if mask[x_a, y_b] == 255 and not flag_autocon:
                        shuxing_mask = ios_point_shuxing_mask(edges, x_a, y_b)
                        if shuxing_mask != 0:
                            edges, x_new, y_new, flag_mask = con_mask(edges, x_a, y_b, mask, hard_distance, shuxing_mask, weight_matrix, conn_mask_valu=20)
                            if flag_mask:
                                flag_autocon = True
                                seedmark_record[np.where(edges == 20)] = seedmark_record[x_a, y_b]
                                new_con_x, new_con_y = np.where(edges == 20)
                                for new_con_idx in range(len(new_con_x)):
                                    nbeibour = pos_neibour_add(nbeibour, new_con_x[new_con_idx], new_con_y[new_con_idx])
                                num = num + len(new_con_x)
                                edges[np.where(edges == 20)] = 1
                                nbeibour = pos_neibour_add(nbeibour, x_new, y_new)
                                shuxing_advance[x_new, y_new] = shuxing_advance[x_a, y_b]
                                positive_advance, shuxing_advance = pos_adv_add(positive_advance, positive, edges, x_new, y_new, shuxing_advance, 0)

                    if not flag_autocon:
                        positive_advance, shuxing_advance = pos_adv_add(positive_advance, positive, edges, x_a, y_b, shuxing_advance, 0)
                    else:
                        positive_advance, shuxing_advance = pos_adv_plus(positive_advance, positive, edges, x_a, y_b,  shuxing_advance, 0)  # 消掉老点


                else:
                    positive_advance, shuxing_advance = pos_adv_add(positive_advance, positive, edges, x_a, y_b, shuxing_advance, 0)

            else:
                positive_advance[x_a, y_b] = threshold_p

    else:
        while (num < pos_need_num):
            x_a, y_b = np.unravel_index(np.argmax(positive_advance), positive_advance.shape)
            if positive_advance[x_a, y_b] <= threshold_p:
                nbeibour_7 = pos_neibour(edges.copy(), kernal=int(min_distance), method='dis')
                positive_cp = positive.copy()
                positive_cp[np.where(edges == 1)] = threshold_p
                positive_cp[np.where(nbeibour_7 > 0)] = threshold_p
                x_a, y_b = np.unravel_index(np.argmax(positive_cp), positive_cp.shape)
                if positive_cp[x_a, y_b] <= threshold_p:
                    break
                positive[x_a, y_b] = threshold_p
                shuxing_advance[x_a, y_b] = 0

            if x_a < pad_num or x_a >= pad_num + height or y_b < pad_num or y_b >= pad_num + width:
                positive_advance[x_a, y_b] = threshold_p
                continue
            if not valit(nbeibour, x_a, y_b):
                positive_advance[x_a, y_b] = threshold_p
                continue
            if edges[x_a, y_b] != 1:
                edges[x_a, y_b] = 1
                seedmark_record, bound_lab_tmp, need_update = seed_mark_add(seedmark_record, x_a, y_b, bound_lab)
                if need_update:
                    bound_lab = bound_lab + 1
                positive_advance[x_a, y_b] = threshold_p
                nbeibour = pos_neibour_add(nbeibour, x_a, y_b)

                if x_a > pad_num and x_a < pad_num + height - 1 and y_b > pad_num and y_b < pad_num + width - 1:
                    positive_advance, shuxing_advance = pos_adv_add(positive_advance, positive, edges, x_a, y_b,
                                                                    shuxing_advance, 0)
                num = num + 1

            else:
                positive_advance[x_a, y_b] = threshold_p




    edges_con = dele_hole(edges.copy())
    edges_con = dele_cornor(edges_con.copy())
    edges_con = connection(edges_con.copy(), int(conne_dis / ratio_decent), weight_matrix)
    edges_con = dele_cornor(edges_con.copy())
    edges_con = dele_bound(edges_con.copy())
    edges_con = dele_cornor(edges_con.copy())
    edges_con = hard_connection(edges_con.copy(), hard_distance)



    pos_num_new = np.where(edges_con[pad_num:height + pad_num, pad_num:width + pad_num] == 1)[0].shape[0]
    num_totla_pos = pos_num_new - pos_num_raw


    if (num_totla_pos < pos_need_num * 0.5 or tune_flag) and epoch_globel > epoch_could_pos_tune:
        flag_t = 'T'
        edges_con[pad_num - 1, :] = 2
        edges_con[pad_num + height, :] = 2
        edges_con[:, pad_num - 1] = 2
        edges_con[:, pad_num + width] = 2
        edges_con = tune_bound(edges_con.copy(), dis_max_fine)


    edges_con[pad_num - 1,:] = -1
    edges_con[pad_num + height,:] = -1
    edges_con[:,pad_num - 1] = -1
    edges_con[:,pad_num + width] = -1
    pos_num_new = np.where(edges_con[pad_num:height + pad_num, pad_num:width + pad_num] == 1)[0].shape[0]
    num_totla_pos = pos_num_new - pos_num_raw


    if (num_totla_pos < pos_need_num / 5 or num_totla_neg < num_cp / 7) and epoch_globel > epoch_could_tune:
        flag_e = 'T'  # 负像素拓展
        edges_con_raw = edges_con.copy()
        edges_con[np.where(edges_con == muhu_value)] = 0
        neg_expand_pad[np.logical_and(edges_con_raw != 0, edges_con == 0)] = 255

    edges_con = edges_con[pad_num:height + pad_num, pad_num:width + pad_num]
    edges_con = (edges_con * 255).astype(np.uint8)
    neg_expand = neg_expand_pad[pad_num:height + pad_num, pad_num:width + pad_num].astype(np.uint8)

    name_new = UNLAB_TRAIN_EDGE_Pseudo_FLIST + '/' + image_name.split('.')[0] + '.png'
    name_new_tmp = UNLAB_TRAIN_EDGE_Pseudo_FLIST + '/' + image_name.split('.')[0] + 'epoch_' + str(epoch_globel) + '_' + flag_t + flag_e + '_' + str(num_totla_pos) + '__' + str(num_totla_neg) + '.png'
    name_new_neg_expand = UNLAB_TRAIN_EDGE_Pseudo_FLIST + '/' + image_name.split('.')[0] + '_neg_expand.png'

    cv2.imwrite(name_new, edges_con)
    cv2.imwrite(name_new_tmp, edges_con)
    cv2.imwrite(name_new_neg_expand, neg_expand)
    print(name_new + ' is OK! :' + str(num_totla_pos) + '__' + str(num_totla_neg))

class bank_fea():
    def __init__(self,num=100,threhold=-10e6,config=None):
        self.fea_medium = None
        self.edges = None
        self.fea_medium_source = None
        self.edges_source = None
        self.num = num
        self.threhold = threhold
        self.config = config

    def reset(self):
        self.fea_medium = None
        self.edges_hot = None
        self.fea_medium_source = None
        self.edges_hot_source = None

    def record_fea(self,edges,fes_medium):
        index = np.random.randint(0, self.config.target_sample)

        fes_medium_ = fes_medium.permute(1, 0, 2, 3)
        fes_medium_ = fes_medium_.reshape([105, fes_medium_.shape[1] * fes_medium_.shape[2] * fes_medium_.shape[3]])[:,index::self.config.target_sample]
        edges_ = edges.permute(1, 0, 2, 3)
        edges_ = edges_.reshape([edges_.shape[1] * edges_.shape[2] * edges_.shape[3]])[index::self.config.target_sample]

        seld_fea_norm = torch.norm(fes_medium_, p=None, dim=0)
        fes_medium_ = fes_medium_ / seld_fea_norm

        fes_medium_np = fes_medium_.cpu().numpy()
        fes_medium_np = fes_medium_np.T
        edges_ = edges_.reshape([1, edges_.shape[0]]).cpu().numpy()
        fes_medium_np_neg = fes_medium_np[np.where(edges_[0] == 0)][::self.config.neg_ample]
        fes_medium_np_pos = fes_medium_np[np.where(edges_[0] == 1)]

        estimator = KMeans(n_clusters=self.config.kmeans_num)  # 构造聚类器
        estimator.fit(fes_medium_np_neg)  # 聚类
        neg_represent = estimator.cluster_centers_

        estimator = KMeans(n_clusters=self.config.kmeans_num)  # 构造聚类器
        estimator.fit(fes_medium_np_pos)  # 聚类
        pos_represent = estimator.cluster_centers_

        fea_medium = np.vstack([neg_represent, pos_represent])
        edges_np = np.ones([1, self.config.kmeans_num + self.config.kmeans_num])
        edges_np[:, :self.config.kmeans_num] = 0

        fea_medium = torch.from_numpy(fea_medium.T).float().cuda()
        edges_np = torch.from_numpy(edges_np).float().cuda()

        if self.edges_hot == None:
            self.edges_hot = edges_np
            self.fea_medium = fea_medium
        else:
            self.edges_hot = torch.cat((self.edges_hot, edges_np), 1)
            self.fea_medium = torch.cat((self.fea_medium, fea_medium), 1)



    def record_fea_source(self,edges,fes_medium):
        index = np.random.randint(0, self.config.source_sample)

        fes_medium_ = fes_medium.permute(1, 0, 2, 3)
        fes_medium_ = fes_medium_.reshape([105, fes_medium_.shape[1] * fes_medium_.shape[2] * fes_medium_.shape[3]])[:,index::self.config.source_sample]
        edges_ = edges.permute(1, 0, 2, 3)
        edges_ = edges_.reshape([edges_.shape[1] * edges_.shape[2] * edges_.shape[3]])[index::self.config.source_sample]

        seld_fea_norm = torch.norm(fes_medium_, p=None, dim=0)
        fes_medium_ = fes_medium_ / seld_fea_norm

        fes_medium_np = fes_medium_.cpu().numpy()
        fes_medium_np = fes_medium_np.T
        edges_ = edges_.reshape([1, edges_.shape[0]]).cpu().numpy()
        fes_medium_np_neg = fes_medium_np[np.where(edges_[0] == 0)][::self.config.neg_ample]
        fes_medium_np_pos = fes_medium_np[np.where(edges_[0] == 1)]

        estimator = KMeans(n_clusters=self.config.kmeans_num)  # 构造聚类器
        estimator.fit(fes_medium_np_neg)  # 聚类
        neg_represent = estimator.cluster_centers_

        estimator = KMeans(n_clusters=self.config.kmeans_num)  # 构造聚类器
        estimator.fit(fes_medium_np_pos)  # 聚类
        pos_represent = estimator.cluster_centers_

        fea_medium = np.vstack([neg_represent, pos_represent])
        edges_np = np.ones([1, self.config.kmeans_num + self.config.kmeans_num])
        edges_np[:, :self.config.kmeans_num] = 0

        fea_medium = torch.from_numpy(fea_medium.T).float().cuda()
        edges_np = torch.from_numpy(edges_np).float().cuda()

        if self.edges_hot_source == None:
            self.edges_hot_source = edges_np
            self.fea_medium_source = fea_medium
        else:
            self.edges_hot_source = torch.cat((self.edges_hot_source, edges_np), 1)
            self.fea_medium_source = torch.cat((self.fea_medium_source, fea_medium), 1)



    def build_label(self,edges, fes_medium_, image_name, epoch):

        fes_medium_ = fes_medium_.permute(1, 0, 2, 3)
        height = fes_medium_.shape[2]
        width = fes_medium_.shape[3]

        if '\\' in image_name[0]:
            if not os.path.exists(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('\\')[-1].replace('.png', '_neg_expand.png')):
                neg_expand = np.zeros([height, width]).astype(np.uint8)
                cv2.imwrite(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('\\')[-1].replace('.png', '_neg_expand.png'), neg_expand)
            else:
                neg_expand = cv2.imread(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('\\')[-1].replace('.png', '_neg_expand.png'), 0)
        else:
            if not os.path.exists(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('/')[-1].replace('.png', '_neg_expand.png')):
                neg_expand = np.zeros([height, width]).astype(np.uint8)
                cv2.imwrite(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('/')[-1].replace('.png', '_neg_expand.png'), neg_expand)
            else:
                neg_expand = cv2.imread(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('/')[-1].replace('.png', '_neg_expand.png'), 0)


        fes_medium_ = fes_medium_.reshape([105, fes_medium_.shape[1] * fes_medium_.shape[2] * fes_medium_.shape[3]])

        fea_norm = torch.norm(fes_medium_, p=None, dim=0)
        fes_medium_ = fes_medium_ / fea_norm


        edges = edges.cpu().numpy()[0,0]
        active_hot_map = fes_medium_.transpose(0, 1).mm(self.fea_medium)
        active_hot_map_neg = torch.max(torch.mul(active_hot_map, 1 - self.edges_hot), dim=1)[0]
        active_hot_map_pos = torch.max(torch.mul(active_hot_map, self.edges_hot), dim=1)[0]

        active_hot_map_source = fes_medium_.transpose(0, 1).mm(self.fea_medium_source)
        active_hot_map_neg_source = torch.max(torch.mul(active_hot_map_source, 1 - self.edges_hot_source), dim=1)[0]
        active_hot_map_pos_source = torch.max(torch.mul(active_hot_map_source, self.edges_hot_source), dim=1)[0]

        active_hot_map_neg = active_hot_map_neg * self.config.arlfa + active_hot_map_neg_source * self.config.beta
        active_hot_map_pos = active_hot_map_pos * self.config.arlfa + active_hot_map_pos_source * self.config.beta

        edges_np = edges.reshape([height * width])
        neg_expand = neg_expand.reshape([height * width])

        index = np.random.randint(0, self.config.test_sample_num)
        fes_medium_unlab = fes_medium_[:,index::self.config.test_sample_num]
        edges_np_unlab = edges_np[index::self.config.test_sample_num].copy()
        neg_expand_unlab = neg_expand[index::self.config.test_sample_num].copy()
        fes_medium_unlab = fes_medium_unlab.detach().cpu().numpy().T

        fes_medium_np_neg = fes_medium_unlab[np.logical_and(edges_np_unlab == 0, neg_expand_unlab == 0)][::self.config.neg_ample]  # 加入了新的neg_expand
        fes_medium_np_pos = fes_medium_unlab[np.where(edges_np_unlab == 1)]

        if fes_medium_np_neg.shape[0] > self.config.kmeans_num:
            estimator = KMeans(n_clusters=self.config.kmeans_num)  # 构造聚类器
            estimator.fit(fes_medium_np_neg)  # 聚类
            neg_represent = estimator.cluster_centers_
        else:
            neg_represent = fes_medium_np_neg

        if fes_medium_np_pos.shape[0] > self.config.kmeans_num:
            estimator = KMeans(n_clusters=self.config.kmeans_num)  # 构造聚类器
            estimator.fit(fes_medium_np_pos)  # 聚类
            pos_represent = estimator.cluster_centers_
        else:
            pos_represent = fes_medium_np_pos

        if len(neg_represent) > 0 and len(pos_represent) > 0:
            fes_medium_np_jihe = np.vstack([neg_represent, pos_represent])
            edges_np_jihe = np.ones([1, fes_medium_np_jihe.shape[0]])
            edges_np_jihe[:, :neg_represent.shape[0]] = 0
            fes_medium_np_jihe = torch.from_numpy(fes_medium_np_jihe.T).float().cuda()
            edges_np_jihe = torch.from_numpy(edges_np_jihe).float().cuda()

            active_hot_map_test = fes_medium_.transpose(0, 1).mm(fes_medium_np_jihe)
            active_hot_map_neg_test = torch.max(torch.mul(active_hot_map_test, 1 - edges_np_jihe), dim=1)[0]
            active_hot_map_pos_test = torch.max(torch.mul(active_hot_map_test, edges_np_jihe), dim=1)[0]

            active_hot_map_neg = active_hot_map_neg + active_hot_map_neg_test * self.config.gema
            active_hot_map_pos = active_hot_map_pos + active_hot_map_pos_test * self.config.gema

        active_final_map = np.vstack([active_hot_map_neg.cpu().numpy(), active_hot_map_pos.cpu().numpy()]).T
        negtive = (active_final_map[:,0]-active_final_map[:,1]).reshape([height,width])
        positive = (active_final_map[:,1]-active_final_map[:,0]).reshape([height,width])


        if '\\' in image_name[0]:
            np.save(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('\\')[-1].replace('.png', '_pos.npy'), positive)  ## lmc for windows
            np.save(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('\\')[-1].replace('.png', '_neg.npy'),negtive)  ## lmc for windows
        else:
            # print(self.config.TEST_EDGE_Pseudo_FLIST + '/' + image_name[0].split('/')[-1].replace('.png', '_pos.npy'), positive)
            np.save(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('/')[-1].replace('.png', '_pos.npy'), positive)  ## lmc for windows
            np.save(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + image_name[0].split('/')[-1].replace('.png', '_neg.npy'),negtive)  ## lmc for windows

        return



class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)
        generator = EdgeGenerator()
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)

        self.bank_fea = bank_fea(self.config.perued_num, self.config.threhold, config=config)

        self.add_module('generator', generator)



        if config.lr_mode == 'ADAM':
            self.gen_optimizer = optim.Adam(params=generator.parameters(),
                                            lr=float(config.LR),
                                            )
        elif config.lr_mode == 'SGD':
            # tune lr
            net_parameters_id = {}
            for pname, p in self.generator.named_parameters():
                if pname in ['conv1_1.weight', 'conv1_2.weight',
                             'conv2_1.weight', 'conv2_2.weight',
                             'conv3_1.weight', 'conv3_2.weight', 'conv3_3.weight',
                             'conv4_1.weight', 'conv4_2.weight', 'conv4_3.weight']:
                    print(pname, 'lr:1 de:1')
                    if 'conv1-4.weight' not in net_parameters_id:
                        net_parameters_id['conv1-4.weight'] = []
                    net_parameters_id['conv1-4.weight'].append(p)
                elif pname in ['conv1_1.bias', 'conv1_2.bias',
                               'conv2_1.bias', 'conv2_2.bias',
                               'conv3_1.bias', 'conv3_2.bias', 'conv3_3.bias',
                               'conv4_1.bias', 'conv4_2.bias', 'conv4_3.bias']:
                    print(pname, 'lr:2 de:0')
                    if 'conv1-4.bias' not in net_parameters_id:
                        net_parameters_id['conv1-4.bias'] = []
                    net_parameters_id['conv1-4.bias'].append(p)
                elif pname in ['conv5_1.weight', 'conv5_2.weight', 'conv5_3.weight']:
                    print(pname, 'lr:100 de:1')
                    if 'conv5.weight' not in net_parameters_id:
                        net_parameters_id['conv5.weight'] = []
                    net_parameters_id['conv5.weight'].append(p)
                elif pname in ['conv5_1.bias', 'conv5_2.bias', 'conv5_3.bias']:
                    print(pname, 'lr:200 de:0')
                    if 'conv5.bias' not in net_parameters_id:
                        net_parameters_id['conv5.bias'] = []
                    net_parameters_id['conv5.bias'].append(p)
                elif pname in ['conv1_1_down.weight', 'conv1_2_down.weight',
                               'conv2_1_down.weight', 'conv2_2_down.weight',
                               'conv3_1_down.weight', 'conv3_2_down.weight', 'conv3_3_down.weight',
                               'conv4_1_down.weight', 'conv4_2_down.weight', 'conv4_3_down.weight',
                               'conv5_1_down.weight', 'conv5_2_down.weight', 'conv5_3_down.weight']:
                    print(pname, 'lr:0.1 de:1')
                    if 'conv_down_1-5.weight' not in net_parameters_id:
                        net_parameters_id['conv_down_1-5.weight'] = []
                    net_parameters_id['conv_down_1-5.weight'].append(p)
                elif pname in ['conv1_1_down.bias', 'conv1_2_down.bias',
                               'conv2_1_down.bias', 'conv2_2_down.bias',
                               'conv3_1_down.bias', 'conv3_2_down.bias', 'conv3_3_down.bias',
                               'conv4_1_down.bias', 'conv4_2_down.bias', 'conv4_3_down.bias',
                               'conv5_1_down.bias', 'conv5_2_down.bias', 'conv5_3_down.bias']:
                    print(pname, 'lr:0.2 de:0')
                    if 'conv_down_1-5.bias' not in net_parameters_id:
                        net_parameters_id['conv_down_1-5.bias'] = []
                    net_parameters_id['conv_down_1-5.bias'].append(p)
                elif pname in ['score_dsn1.weight', 'score_dsn2.weight', 'score_dsn3.weight',
                               'score_dsn4.weight', 'score_dsn5.weight',
                               'score_dsn1s.weight', 'score_dsn2s.weight', 'score_dsn3s.weight',
                               'score_dsn4s.weight', 'score_dsn5s.weight']:
                    print(pname, 'lr:0.01 de:1')
                    if 'score_dsn_1-5.weight' not in net_parameters_id:
                        net_parameters_id['score_dsn_1-5.weight'] = []
                    net_parameters_id['score_dsn_1-5.weight'].append(p)
                elif pname in ['score_dsn1.bias', 'score_dsn2.bias', 'score_dsn3.bias',
                               'score_dsn4.bias', 'score_dsn5.bias',
                               'score_dsn1s.bias', 'score_dsn2s.bias', 'score_dsn3s.bias',
                               'score_dsn4s.bias', 'score_dsn5s.bias']:
                    print(pname, 'lr:0.02 de:0')
                    if 'score_dsn_1-5.bias' not in net_parameters_id:
                        net_parameters_id['score_dsn_1-5.bias'] = []
                    net_parameters_id['score_dsn_1-5.bias'].append(p)
                elif pname in ['score_final.weight', 'score_finals.weight']:
                    print(pname, 'lr:0.001 de:1')
                    if 'score_final.weight' not in net_parameters_id:
                        net_parameters_id['score_final.weight'] = []
                    net_parameters_id['score_final.weight'].append(p)
                elif pname in ['score_final.bias', 'score_finals.bias']:
                    print(pname, 'lr:0.002 de:0')
                    if 'score_final.bias' not in net_parameters_id:
                        net_parameters_id['score_final.bias'] = []
                    net_parameters_id['score_final.bias'].append(p)

            self.gen_optimizer = torch.optim.SGD([
                {'params': net_parameters_id['conv1-4.weight'], 'lr': config.LR * 1, 'weight_decay': config.lr_weight_decay},
                {'params': net_parameters_id['conv1-4.bias'], 'lr': config.LR * 2, 'weight_decay': 0.},
                {'params': net_parameters_id['conv5.weight'], 'lr': config.LR * 100, 'weight_decay': config.lr_weight_decay},
                {'params': net_parameters_id['conv5.bias'], 'lr': config.LR * 200, 'weight_decay': 0.},
                {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': config.LR * 0.1,
                 'weight_decay': config.lr_weight_decay},
                {'params': net_parameters_id['conv_down_1-5.bias'], 'lr': config.LR * 0.2, 'weight_decay': 0.},
                {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': config.LR * 0.01,
                 'weight_decay': config.lr_weight_decay},
                {'params': net_parameters_id['score_dsn_1-5.bias'], 'lr': config.LR * 0.02, 'weight_decay': 0.},
                {'params': net_parameters_id['score_final.weight'], 'lr': config.LR * 0.001,
                 'weight_decay': config.lr_weight_decay},
                {'params': net_parameters_id['score_final.bias'], 'lr': config.LR * 0.002, 'weight_decay': 0.},
            ], lr=config.LR, momentum=config.lr_momentum, weight_decay=config.lr_weight_decay)


        elif config.lr_mode == 'SGD_normal':
            self.gen_optimizer = optim.SGD(params=generator.parameters(),
                                       lr=float(config.LR),
                                       # betas=(config.BETA1, config.BETA2)
                                       )
        self.weight = 0.95


    def medium_feature(self, images, edges, images_name = None,  mode=1, tune_flag = False, epoch = 1000):

        # process outputs
        _, _, fes_medium = self(images)

        if mode == 1:
            self.bank_fea.record_fea(edges.detach(), fes_medium.detach()) #edge_lab_fuzzy
        elif mode == 2:
            self.bank_fea.record_fea_source(edges.detach(), fes_medium.detach())  #edge_lab_fuzzy
        else:
            self.bank_fea.build_label(edges.detach(), fes_medium.detach(), images_name, epoch = self.iteration)  #edge_lab_fuzzy

        return

    def process(self, images, edges, mode=1):  # 0: eval 1: one-shot 2: source 3：test dataa

        # process outputs
        outputs, source_outputs, fes_medium = self(images)

        if mode == 2:
            outputs = source_outputs

        gen_bce_loss = 0
        for idxx,o in enumerate(outputs):
            if idxx == 5:
                output = o.detach()
            if mode == 1:
                loss_tmp, _ = cross_entropy_loss_RCF(o, edges)
            elif mode == 2:
                loss_tmp, _ = cross_entropy_loss_RCF(o, edges)
            else:
                loss_tmp = cross_entropy_loss_perued_RCF(o, edges)
            gen_bce_loss += loss_tmp

        # gen bce loss
        gen_loss = gen_bce_loss

        # create logs
        logs = [
            ("l_bce", gen_loss.item()),
        ]
        return output, gen_loss, logs

    def forward(self, images):
        inputs = images
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None):
        if gen_loss is not None:
            gen_loss.backward()

    def step(self):
        self.gen_optimizer.step()
        self.gen_optimizer.zero_grad()