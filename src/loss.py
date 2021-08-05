import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.nn import functional as FF
import numpy as np
from PIL import Image


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

def cross_entropy_loss_RCF(outputs, label):
    mask = torch.torch.zeros(label.shape[0], label.shape[1], label.shape[2], label.shape[3]).cuda()
    num_positive = torch.sum((label == 1).float()).float()
    num_negative = torch.sum((label == 0).float()).float()
    if num_positive + num_negative > 0:
        mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[label == 0] = 1.1 * num_positive / (num_positive + num_negative)


    cost = torch.nn.functional.binary_cross_entropy(
        outputs.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost), 1.0 * num_negative / (num_positive + num_negative)

def cross_entropy_loss_perued_RCF(outputs, label):

    num_positive = torch.sum((label == 1).float()).float()
    num_negative = torch.sum((label == 0).float()).float()
    num_fuzzy = label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3]

    if num_positive + num_negative > 0:
        '''
        mask = torch.torch.ones(label.shape[0], label.shape[1], label.shape[2], label.shape[3]).cuda()
        neg_weight = num_positive / (num_positive + num_negative)
        if num_fuzzy - num_negative - num_positive > 0:
            fuz_weight_ = num_positive * num_negative / (num_positive + num_negative) / (num_fuzzy - num_negative - num_positive)
        else:
            fuz_weight_ = neg_weight / 2
        mask = mask * fuz_weight_
        '''
        mask = torch.torch.zeros(label.shape[0], label.shape[1], label.shape[2], label.shape[3]).cuda()
        # mask = torch.torch.zeros(label.shape[0], label.shape[1], label.shape[2], label.shape[3]).cuda()
        mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[label == 0] = 1.1 * num_positive / (num_positive + num_negative)

    else:
        mask = torch.torch.zeros(label.shape[0], label.shape[1], label.shape[2], label.shape[3]).cuda()
    cost = torch.nn.functional.binary_cross_entropy(outputs.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss