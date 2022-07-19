import sys
sys.path.append("..")
from loss import *
import torch
import math
import time
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
class SSIM(nn.Module):
    def __init__(self, size_average=False, max_val=255, channel=3):
        super(SSIM, self).__init__()
        self.size_average = size_average
        self.channel = channel
        self.max_val = max_val

    def ssim(self, img1, img2):
        size_average = self.size_average
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=self.channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=self.channel) - mu1_mu2
        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        # mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean()
            # return ssim_map.mean(), mcs_map.mean()
        else:
            return ssim_map
            # return ssim_map, mcs_map

    def forward(self, img1, img2):
        ssim = self.ssim(img1, img2)
        return 1 - ssim


class train_loss_net(nn.Module):
    def __init__(self, pixel_loss, re_weight, channel=3):
        super(train_loss_net, self).__init__()
        if pixel_loss == 'MSE':
            self.pixel_loss = MSE(size_average=True)
        else:
            self.pixel_loss = MAE(size_average=True)
        self.ssim = SSIM(size_average=True, max_val=1, channel=channel)
        self.VGG_LOSS = VGG_LOSS(size_average=True)
        self.re_weight = re_weight

    def forward(self, img, weight):
        # 计算逐像素损失图
        J, gt_image = img
        loss_for_train = 0
        loss = [0] * len(weight)
        loss[0] = self.pixel_loss(J, gt_image)
        loss[1] = self.ssim(J, gt_image)
        loss[2] = self.VGG_LOSS(J, gt_image)
        # image_weight = self.pixel_loss(re_image, gt_image)
        # loss[0] = torch.mean(torch.pow(self.re_weight, image_weight) * self.pixel_loss(J, gt_image))
        for i in range(len(weight)):
            loss_for_train = loss_for_train + loss[i] * weight[i]
            loss[i] = loss[i].item()
        return loss_for_train, loss


class test_loss_net(nn.Module):
    def __init__(self, pixel_loss, re_weight, channel=3):
        super(test_loss_net, self).__init__()
        if pixel_loss == 'MSE':
            self.pixel_loss = MSE()
        else:
            self.pixel_loss = MAE()
        self.ssim = SSIM(size_average=False, data_range=1, channel=channel)
        self.VGG_LOSS = VGG_LOSS()
        self.re_weight = re_weight

    def forward(self, img, weight):
        # 计算逐像素损失图
        J, gt_image, re_image = img
        loss_for_train = 0
        loss = [0] * len(weight)
        image_weight = self.pixel_loss(re_image, gt_image)
        loss[0] = torch.mean(torch.pow(self.re_weight, image_weight) * self.pixel_loss(J, gt_image))
        for i in range(len(weight)):
            loss[i] = loss[i].item()
        return loss


class fiveK_loss(nn.Module):
    def __init__(self, pixel_loss, channel=3):
        super(fiveK_loss, self).__init__()
        if pixel_loss == 'MSE':
            self.pixel_loss = MSE(size_average=True)
        else:
            self.pixel_loss = MAE(size_average=True)
        self.ssim = MS_SSIM(data_range=1, size_average=True, channel=channel)
        self.VGG_LOSS = VGG(size_average=True)

    def forward(self, img, weight):
        # 计算逐像素损失图
        J, gt_image = img
        loss_for_train = 0
        loss = [0] * len(weight)
        loss[0] = self.pixel_loss(J, gt_image)
        loss[1] = 1 - self.ssim(J, gt_image)
        loss[2] = self.VGG_LOSS(J, gt_image)
        for i in range(len(weight)):
            loss_for_train = loss_for_train + loss[i] * weight[i]
            loss[i] = loss[i].item()
        return loss_for_train, loss
