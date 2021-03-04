import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread #imageio
#from imageio import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask
import matplotlib.pyplot as plt

from PIL import ImageFilter
#Dataset_unlabel


def trans_img_0_255(img, mode):
    if mode =='rgb':
        img = img[:, :, ::-1] - np.zeros_like(img)  # rgb to bgr
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float() * 255
        img_t[0, :, :] = img_t[0, :, :] - 104.00698793
        img_t[1, :, :] = img_t[1, :, :] - 116.66876762
        img_t[2, :, :] = img_t[2, :, :] - 122.67891434
        return img_t
    elif mode == 'gray':
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t


class Dataset_unlabel(torch.utils.data.Dataset):
    def __init__(self, config, flist,augment=True, training=True, mode = 'gray'):
        super(Dataset_unlabel, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.input_size = config.INPUT_SIZE
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)



        '''
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(img_gray,cmap=plt.cm.gray)
        plt.subplot(132)
        plt.imshow(msak,cmap=plt.cm.gray)
        plt.subplot(133)
        plt.imshow(edge,cmap=plt.cm.gray)
        plt.show()
        '''
        #return self.to_tensor(img), self.to_tensor(img_gray),self.to_tensor(edge), self.fuzzy_to_tensor(edge,msak), self.balance_to_tensor(edge),self.to_tensor(msak*255)
        # 在金相数据集上mask=1 而bsds相反
        return trans_img_0_255(img, self.mode)


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t


    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, msak_flist,augment=True, training=True, mode = 'gray'):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.msak_data = self.load_flist(msak_flist)
        self.input_size = config.INPUT_SIZE
        self.data_dir = flist
        self.gt_dir = edge_flist
        self.msak_dir = msak_flist
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        #index = 0  # delete it
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)

        # load edge
        gt_name = self.data[index].replace(self.data_dir, self.gt_dir)
        edge = self.load_edge(img_gray, gt_name)


        # load mask
        mask_name = self.data[index].replace(self.data_dir, self.msak_dir)
        msak = self.load_mask(img_gray, mask_name)


        #return self.to_tensor(img), self.to_tensor(img_gray),self.to_tensor(edge), self.fuzzy_to_tensor(edge,msak), self.balance_to_tensor(edge),self.to_tensor(msak*255)
        # 在金相数据集上mask=1 而bsds相反
        return trans_img_0_255(img, self.mode), self.to_tensor(edge), self.to_tensor(msak * 255)
            # edge), self.to_tensor(255 - msak * 255)

    def load_edge(self, img, gt_name):

        imgh, imgw = img.shape[0:2]
        edge = imread(gt_name)
        if len(edge.shape) == 3:
            edge = edge[:,:,0]
        edge = self.resize(edge, imgh, imgw)

        return edge

    def load_ex_edge(self, img, index):

        imgh, imgw = img.shape[0:2]
        edge = imread(self.ex_edge[index])
        edge = self.resize(edge, imgh, imgw)

        return edge

    def load_mask(self, img, mask_name):
        imgh, imgw = img.shape[0:2]
        mask = imread(mask_name)
        mask = self.resize(mask, imgh, imgw)
        mask[np.where(mask>0)] = 1
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def balance_to_tensor(self,img):
        img = Image.fromarray(img)
        #im_blur = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        img_ = np.array(img)
        neg = np.where(img_==0)[0].shape[0]
        pos = img_.shape[0] * img_.shape[1] - neg
        bal = np.zeros(img_.shape)

        #bal[np.where(img_==0)] = np.sqrt(pos/(pos+neg))
        #bal[np.where(img_>0)] = np.sqrt(neg/(pos+neg))
        bal[np.where(img_==0)] = pos/(pos+neg)
        bal[np.where(img_>0)] = neg/(pos+neg)

        #ssss = bal.copy()
        bal = Image.fromarray(bal)
        '''
        img_raw =img_.copy()
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(bal)
        plt.show()
        '''
        img_t = F.to_tensor(bal).float()
        return img_t

    def balance_to_mask(self,img):
        img = Image.fromarray(img)
        #im_blur = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        img_ = np.array(img)
        neg = np.where(img_==0)[0].shape[0]
        pos = img_.shape[0] * img_.shape[1] - neg
        bal = np.zeros(img_.shape)

        bal[np.where(img_==0)] = 1#np.sqrt(pos/(pos+neg)) 0.8
        bal[np.where(img_>0)] = 1#np.sqrt(neg/(pos+neg)) 0.6


        #ssss = bal.copy()
        bal = Image.fromarray(bal)
        '''
        img_raw =img_.copy()
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(bal)
        plt.show()
        '''
        img_t = F.to_tensor(bal).float()
        return img_t

    def fuzzy_to_tensor(self, img, mask):
        img = Image.fromarray(img)
        im_blur = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        img_ = np.array(img)
        img__ = img_.copy()
        im_blur_ = np.array(im_blur)
        img_[np.where(mask==0)[0],np.where(mask==0)[1]] = im_blur_[np.where(mask==0)[0],np.where(mask==0)[1]]
        img_[np.where(img__>0)[0],np.where(img__>0)[1]] = 255
        img_ = Image.fromarray(img_)
        '''
        img_raw =img_.copy()
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(img_)
        plt.subplot(133)
        plt.imshow(mask)
        plt.show()
        '''
        img_t = F.to_tensor(img_).float()
        return img_t

    def fuzzy_lab_tensor(self, img):
        img = Image.fromarray(img)
        im_blur = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        img_ = np.array(img)
        img__ = img_.copy()
        im_blur_ = np.array(im_blur)
        img_ = im_blur_
        img_[np.where(img_>0)[0],np.where(img_>0)[1]] = 255
        img_ = Image.fromarray(img_)
        '''
        img_raw =img_.copy()
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(img_)
        plt.subplot(133)
        plt.imshow(mask)
        plt.show()
        '''
        img_t = F.to_tensor(img_).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


class Dataset_out(torch.utils.data.Dataset):
    def __init__(self, config, flist, augment=True, training=True,  mode = 'gray'):
        super(Dataset_out, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.input_size = config.INPUT_SIZE
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)

        #return self.to_tensor(img), self.to_tensor(img_gray),self.to_tensor(edge), self.fuzzy_to_tensor(edge,msak), self.balance_to_tensor(edge),self.to_tensor(msak*255)
        # 在金相数据集上mask=1 而bsds相反
        return trans_img_0_255(img, self.mode)


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class Dataset_semi(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, msak_flist,augment=True, training=True,  mode = 'gray'):
        super(Dataset_semi, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.msak_data = self.load_flist(msak_flist)
        self.input_size = config.INPUT_SIZE
        self.data_dir = flist
        self.gt_dir = edge_flist
        self.msak_dir = msak_flist
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        #index = 0  # delete it
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)

        # load edge
        gt_name = self.data[index].replace(self.data_dir, self.gt_dir)
        edge = self.load_edge(img_gray, gt_name)


        # load mask
        mask_name = self.data[index].replace(self.data_dir, self.msak_dir)
        msak = self.load_mask(img_gray, mask_name)


        #return self.to_tensor(img), self.to_tensor(img_gray),self.to_tensor(edge), self.fuzzy_to_tensor(edge,msak), self.balance_to_tensor(edge),self.to_tensor(msak*255)
        # 在金相数据集上mask=1 而bsds相反
        return trans_img_0_255(img, self.mode), self.to_tensor(edge), self.to_tensor(msak * 255), self.data[index]
            # edge), self.to_tensor(255 - msak * 255)

    def load_edge(self, img, gt_name):

        imgh, imgw = img.shape[0:2]
        edge = imread(gt_name)
        if len(edge.shape) == 3:
            edge = edge[:,:,0]
        edge = self.resize(edge, imgh, imgw)

        return edge

    def load_ex_edge(self, img, mask_name):

        imgh, imgw = img.shape[0:2]
        edge = imread(mask_name)
        edge = self.resize(edge, imgh, imgw)

        return edge

    def load_mask(self, img, mask_name):
        imgh, imgw = img.shape[0:2]
        mask = imread(mask_name)
        mask = self.resize(mask, imgh, imgw)
        mask[np.where(mask>0)] = 1
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def balance_to_tensor(self,img):
        img = Image.fromarray(img)
        #im_blur = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        img_ = np.array(img)
        neg = np.where(img_==0)[0].shape[0]
        pos = img_.shape[0] * img_.shape[1] - neg
        bal = np.zeros(img_.shape)

        #bal[np.where(img_==0)] = np.sqrt(pos/(pos+neg))
        #bal[np.where(img_>0)] = np.sqrt(neg/(pos+neg))
        bal[np.where(img_==0)] = pos/(pos+neg)
        bal[np.where(img_>0)] = neg/(pos+neg)

        #ssss = bal.copy()
        bal = Image.fromarray(bal)
        '''
        img_raw =img_.copy()
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(bal)
        plt.show()
        '''
        img_t = F.to_tensor(bal).float()
        return img_t

    def balance_to_mask(self,img):
        img = Image.fromarray(img)
        #im_blur = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        img_ = np.array(img)
        neg = np.where(img_==0)[0].shape[0]
        pos = img_.shape[0] * img_.shape[1] - neg
        bal = np.zeros(img_.shape)

        bal[np.where(img_==0)] = 1#np.sqrt(pos/(pos+neg)) 0.8
        bal[np.where(img_>0)] = 1#np.sqrt(neg/(pos+neg)) 0.6


        #ssss = bal.copy()
        bal = Image.fromarray(bal)
        '''
        img_raw =img_.copy()
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(bal)
        plt.show()
        '''
        img_t = F.to_tensor(bal).float()
        return img_t

    def fuzzy_to_tensor(self, img, mask):
        img = Image.fromarray(img)
        im_blur = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        img_ = np.array(img)
        img__ = img_.copy()
        im_blur_ = np.array(im_blur)
        img_[np.where(mask==0)[0],np.where(mask==0)[1]] = im_blur_[np.where(mask==0)[0],np.where(mask==0)[1]]
        img_[np.where(img__>0)[0],np.where(img__>0)[1]] = 255
        img_ = Image.fromarray(img_)
        '''
        img_raw =img_.copy()
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(img_)
        plt.subplot(133)
        plt.imshow(mask)
        plt.show()
        '''
        img_t = F.to_tensor(img_).float()
        return img_t

    def fuzzy_lab_tensor(self, img):
        img = Image.fromarray(img)
        im_blur = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        img_ = np.array(img)
        img__ = img_.copy()
        im_blur_ = np.array(im_blur)
        img_ = im_blur_
        img_[np.where(img_>0)[0],np.where(img_>0)[1]] = 255
        img_ = Image.fromarray(img_)
        '''
        img_raw =img_.copy()
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(img_)
        plt.subplot(133)
        plt.imshow(mask)
        plt.show()
        '''
        img_t = F.to_tensor(img_).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item