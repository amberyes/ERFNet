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
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class fiveK_loss_test(nn.Module):
    def __init__(self, channel=3):
        super(fiveK_loss, self).__init__()
        self.pixel_loss = MAE(size_average=True)
        self.ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=channel)
        self.vgg_loss = VGG(size_average=True)

    def forward(self,Jf, gt_image):
        # 计算逐像素损失图
        # Jf, gt_image = img #Jre, Jen,
        # loss = [0] * len(weight)
        loss_for_train = 0

        #
        # loss[0] = self.pixel_loss(Jre, gt_image)
        # loss[1] = 1 - self.ssim_loss(Jre, gt_image)
        # loss[2] = self.vgg_loss(Jre, gt_image)
        #
        # loss[3] = self.pixel_loss(Jen, gt_image)
        # loss[4] = 1 - self.ssim_loss(Jen, gt_image)
        # loss[5] = self.vgg_loss(Jen, gt_image)

        # loss[2] = torch.mean(torch.min(loss[0], loss[1]))
        # loss[0] = torch.mean(loss[0])
        # loss[1] = torch.mean(loss[1])
        loss = [0] * len(Jf)
        loss[0] = self.pixel_loss(Jf, gt_image)
        loss[1] = 1 - self.ssim_loss(Jf, gt_image)
        loss[2] = self.vgg_loss(Jf, gt_image)

        # for i in range(len(weight)):
        #     loss_for_train = loss_for_train + loss[i] * weight[i]
        #     loss[i] = loss[i].item()
        return  loss #loss_for_train,

class train_loss_wAt(nn.Module):
    def __init__(self, channel=3, pixel_loss='MSE'):
        super(train_loss_net, self).__init__()
        if pixel_loss == 'MSE':
            self.pixel = MSE(size_average=False)
        else:
            self.pixel = MAE(size_average=False)
        self.ssim = MS_SSIM(size_average=True, data_range=1, channel=channel)
        self.vgg = VGG(size_average=True)

    def forward(self, img, weight):
        # 计算逐像素损失图
        A, A_gt, t, t_gt, Jre, Jen, Jf, gt_image = img
        loss = [0] * len(weight)
        loss_for_train = 0

        loss[0] = self.pixel_loss(Jre, gt_image)
        loss[1] = self.pixel_loss(Jen, gt_image)
        loss[2] = torch.mean(torch.min(loss[0], loss[1]))
        loss[0] = torch.mean(loss[0])
        loss[1] = torch.mean(loss[1])
        loss[3] = torch.mean(self.pixel_loss(Jf, gt_image))
        loss[4] = 1 - self.ssim(Jf, gt_image)

        for i in range(len(weight)):
            loss_for_train = loss_for_train + loss[i]
            loss[i] = loss[i].item()
        return loss_for_train, loss


class fiveK_loss(nn.Module):
    def __init__(self, channel=3):
        super(fiveK_loss, self).__init__()
        self.pixel_loss = MAE(size_average=True)
        self.ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=channel)
        self.vgg_loss = VGG(size_average=True)

    def forward(self, img, weight):
        # 计算逐像素损失图
        Jre, Jen, Jf, gt_image = img
        loss = [0] * len(weight)
        loss_for_train = 0

        loss[0] = self.pixel_loss(Jre, gt_image)
        loss[1] = 1 - self.ssim_loss(Jre, gt_image)
        loss[2] = self.vgg_loss(Jre, gt_image)

        loss[3] = self.pixel_loss(Jen, gt_image)
        loss[4] = 1 - self.ssim_loss(Jen, gt_image)
        loss[5] = self.vgg_loss(Jen, gt_image)

        # loss[2] = torch.mean(torch.min(loss[0], loss[1]))
        # loss[0] = torch.mean(loss[0])
        # loss[1] = torch.mean(loss[1])

        loss[6] = self.pixel_loss(Jf, gt_image)
        loss[7] = 1 - self.ssim_loss(Jf, gt_image)
        loss[8] = self.vgg_loss(Jf, gt_image)

        for i in range(len(weight)):
            loss_for_train = loss_for_train + loss[i] * weight[i]
            loss[i] = loss[i].item()
        return loss_for_train, loss


class test_loss_net_2(nn.Module):
    def __init__(self, weight, size_average=True, channel=3):
        super(test_loss_net_2, self).__init__()
        self.pixel_loss = MAE(size_average=size_average)
        self.ssim = SSIM(size_average=size_average, data_range=1, channel=channel)
        self.VGG_LOSS = VGG_LOSS(size_average=size_average)
        self.weight = weight

    def forward(self, dehazy, gt):
        # 计算逐像素损失
        loss_for_save = [0] * len(dehazy) * 3
        pixel_loss = [0] * len(dehazy)
        for i in range(len(dehazy)):
            pixel_loss[i] = self.pixel_loss(dehazy, gt)
            loss_for_save[i] = pixel_loss[i].item()
            # print(pixel_loss[i].shape)

        # 计算ssim损失
        ssim_loss = [0] * len(dehazy)
        for i in range(len(dehazy)):
            ssim_loss[i] = self.ssim(dehazy, gt)
            loss_for_save[i + len(dehazy)] = ssim_loss[i].item()

        # 计算vgg损失
        vgg_loss = [0] * len(dehazy)
        for i in range(len(dehazy)):
            vgg_loss[i] = self.VGG_LOSS(dehazy, gt)
            loss_for_save[i + len(dehazy) * 2] = vgg_loss[i].item()

        return loss_for_save


class test_loss_net_1(nn.Module):
    def __init__(self, weight, size_average=True, channel=3):
        super(test_loss_net_1, self).__init__()
        self.pixel_loss = MAE(size_average=size_average)
        self.ssim = SSIM(size_average=size_average, data_range=1, channel=channel)
        self.VGG_LOSS = VGG_LOSS(size_average=size_average)
        self.weight = weight

    def forward(self, dehazy, gt):
        # 计算逐像素损失
        # loss_for_save = [0] * 3 * (len(dehazy) * 2 - 1)
        # pixel_loss = [0] * len(dehazy)
        # print(dehazy[0].size())
        # print(gt.size())
        temp = [0] * len(dehazy)
        for i in range(len(dehazy)):
            # pixel_loss[i] = self.pixel_loss(dehazy[i], gt)
            temp[i] = self.pixel_loss(dehazy[i], gt).item()
        loss_for_save = temp

        temp = [0] * len(dehazy)
        for i in range(len(dehazy)):
            temp[i] = self.ssim(dehazy[i], gt).item()
        loss_for_save += temp

        temp = [0] * len(dehazy)
        for i in range(len(dehazy)):
            temp[i] = self.VGG_LOSS(dehazy[i], gt).item()
        loss_for_save += temp

        # print(pixel_loss[i].shape)
        temp = [0] * (len(dehazy) - 1)
        for i in range(len(dehazy) - 1):
            temp[i] = self.pixel_loss(dehazy[i], dehazy[i + 1]).item()
        loss_for_save += temp
        # (len(loss_for_save))

        temp = [0] * (len(dehazy) - 1)
        for i in range(len(dehazy) - 1):
            temp[i] = self.ssim(dehazy[i], dehazy[i + 1]).item()
        loss_for_save += temp

        temp = [0] * (len(dehazy) - 1)
        for i in range(len(dehazy) - 1):
            temp[i] = self.VGG_LOSS(dehazy[i], dehazy[i + 1]).item()
        loss_for_save += temp

        return loss_for_save


class test_loss_woAt(nn.Module):
    def __init__(self):
        super(test_loss_woAt, self).__init__()
        self.pixel_loss = MAE(size_average=False)
        self.ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=3)
        self.vgg_loss = VGG_LOSS(size_average=True)

    def forward(self, img, weight):
        # 计算逐像素损失图
        Jre, Jen, Jf, gt_image = img
        loss = [0] * len(weight)
        loss_for_train = 0
        loss[0] = self.pixel_loss(Jre, gt_image)
        loss[1] = self.pixel_loss(Jen, gt_image)
        loss[2] = torch.mean(torch.min(loss[0], loss[1]))
        loss[0] = torch.mean(loss[0])
        loss[1] = torch.mean(loss[1])
        loss[3] = torch.mean(self.pixel_loss(Jf, gt_image))
        for i in range(len(weight)):
            loss[i] = loss[i].item()
        return loss
