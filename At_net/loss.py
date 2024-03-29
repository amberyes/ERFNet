import torch
import math
import time
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


class MAE(nn.Module):
    def __init__(self, size_average=False):
        super(MAE, self).__init__()
        self.size_average = size_average

    def forward(self, x1, x2):
        x1 = x1 - x2
        x1 = abs(x1)
        if self.size_average:
            return x1.mean()
        else:
            return x1


class MSE(nn.Module):
    def __init__(self, size_average=False):
        super(MSE, self).__init__()
        self.size_average = size_average

    def forward(self, x1, x2):
        x1 = x1 - x2
        x1 = x1.pow(2)
        if self.size_average:
            return x1.mean()
        else:
            return x1


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
        return out


'''
    def ms_ssim(self, img1, img2, levels=5):
        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())
        msssim = Variable(torch.Tensor(levels, ).cuda())
        mcs = Variable(torch.Tensor(levels, ).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2
        value = (torch.prod(mcs[0:levels - 1] ** weight[0:levels - 1]) *
                 (msssim[levels - 1] ** weight[levels - 1]))
        return value
'''


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


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


class VGG_LOSS(nn.Module):
    def __init__(self, size_average=False):
        super(VGG_LOSS, self).__init__()
        # self.vgg = Vgg16().type(torch.cuda.FloatTensor)
        self.vgg = VGG16()
        self.l2 = MSE(size_average)
        self.size_average = size_average

    def forward(self, hazy, gth):
        output_features = self.vgg(hazy)
        gth_features = self.vgg(gth)
        out_1 = self.l2(output_features[0], gth_features[0])
        out_2 = self.l2(output_features[1], gth_features[1])
        out_3 = self.l2(output_features[2], gth_features[2])
        out_4 = self.l2(output_features[3], gth_features[3])
        if self.size_average:
            return (out_1 + out_2 + out_3 + out_4) / 4
        else:
            return [out_1, out_2, out_3, out_4]


class train_loss_net(nn.Module):
    def __init__(self, channel=3, pixel_loss='MSE'):
        super(train_loss_net, self).__init__()
        if pixel_loss == 'MSE':
            self.pixel_loss = MSE(size_average=True)
        else:
            self.pixel_loss = MAE(size_average=True)
        self.ssim = SSIM(size_average=True, max_val=1, channel=channel)
        self.VGG_LOSS = VGG_LOSS(size_average=True)

    def forward(self, img, weight):
        # 计算逐像素损失图
        A, A_gth, t, t_gth, J_reconstruct, gt_image = img
        loss = [0] * len(weight)
        loss_for_train = 0
        loss[0] = self.pixel_loss(A, A_gth)
        loss[1] = self.pixel_loss(t, t_gth)
        loss[2] = self.ssim(t, t_gth)
        loss[3] = self.pixel_loss(J_reconstruct, gt_image)
        loss[4] = self.ssim(J_reconstruct, gt_image)
        loss[5] = self.VGG_LOSS(J_reconstruct, gt_image)
        for i in range(len(weight)):
            loss_for_train = loss_for_train + loss[i]
            loss[i] = loss[i].item()
        return loss_for_train, loss


class test_loss_net_2(nn.Module):
    def __init__(self, weight, size_average=True, channel=3):
        super(test_loss_net_2, self).__init__()
        self.pixel_loss = MAE(size_average=size_average)
        self.ssim = SSIM(size_average=size_average, max_val=1, channel=channel)
        self.VGG_LOSS = VGG_LOSS(size_average=size_average)
        self.weight = weight

    def forward(self, dehazy, gth):
        # 计算逐像素损失
        loss_for_save = [0] * len(dehazy) * 3
        pixel_loss = [0] * len(dehazy)
        for i in range(len(dehazy)):
            pixel_loss[i] = self.pixel_loss(dehazy, gth)
            loss_for_save[i] = pixel_loss[i].item()
            # print(pixel_loss[i].shape)

        # 计算ssim损失
        ssim_loss = [0] * len(dehazy)
        for i in range(len(dehazy)):
            ssim_loss[i] = self.ssim(dehazy, gth)
            loss_for_save[i + len(dehazy)] = ssim_loss[i].item()

        # 计算vgg损失
        vgg_loss = [0] * len(dehazy)
        for i in range(len(dehazy)):
            vgg_loss[i] = self.VGG_LOSS(dehazy, gth)
            loss_for_save[i + len(dehazy) * 2] = vgg_loss[i].item()

        return loss_for_save


class test_loss_net_1(nn.Module):
    def __init__(self, weight, size_average=True, channel=3):
        super(test_loss_net_1, self).__init__()
        self.pixel_loss = MAE(size_average=size_average)
        self.ssim = SSIM(size_average=size_average, max_val=1, channel=channel)
        self.VGG_LOSS = VGG_LOSS(size_average=size_average)
        self.weight = weight

    def forward(self, dehazy, gth):
        # 计算逐像素损失
        # loss_for_save = [0] * 3 * (len(dehazy) * 2 - 1)
        # pixel_loss = [0] * len(dehazy)
        # print(dehazy[0].size())
        # print(gth.size())
        temp = [0] * len(dehazy)
        for i in range(len(dehazy)):
            # pixel_loss[i] = self.pixel_loss(dehazy[i], gth)
            temp[i] = self.pixel_loss(dehazy[i], gth).item()
        loss_for_save = temp

        temp = [0] * len(dehazy)
        for i in range(len(dehazy)):
            temp[i] = self.ssim(dehazy[i], gth).item()
        loss_for_save += temp

        temp = [0] * len(dehazy)
        for i in range(len(dehazy)):
            temp[i] = self.VGG_LOSS(dehazy[i], gth).item()
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


class test_loss_nogt(nn.Module):
    def __init__(self, channel=3, pixel_loss='MSE'):
        super(test_loss_nogt, self).__init__()
        if pixel_loss == 'MSE':
            self.pixel_loss = MSE(size_average=True)
        else:
            self.pixel_loss = MAE(size_average=True)
        self.ssim = SSIM(size_average=True, max_val=1, channel=channel)
        self.VGG_LOSS = VGG_LOSS(size_average=True)

    def forward(self, img, weight):
        # 计算逐像素损失图
        J_reconstruct, gt_image = img
        loss = [0] * len(weight)
        loss[0] = self.pixel_loss(J_reconstruct, gt_image).item()
        loss[1] = self.ssim(J_reconstruct, gt_image).item()
        loss[2] = self.VGG_LOSS(J_reconstruct, gt_image).item()
        return loss
