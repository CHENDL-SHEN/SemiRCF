import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.nn import functional as FF
import numpy as np
from PIL import Image




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

