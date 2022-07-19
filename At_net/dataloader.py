from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as f
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio
import torch
import random
import math
from PIL import Image


# 实现旋转，反转。一共8种形态。
def data_aug(img):
    a = random.random()
    b = math.floor(random.random() * 4)
    if a >= 0.5:
        for i in range(len(img)):
            img[i] = img[i].transpose(Image.FLIP_LEFT_RIGHT)
    if b == 1:
        for i in range(len(img)):
            img[i] = img[i].transpose(Image.ROTATE_90)
    elif b == 2:
        for i in range(len(img)):
            img[i] = img[i].transpose(Image.ROTATE_180)
    elif b == 3:
        for i in range(len(img)):
            img[i] = img[i].transpose(Image.ROTATE_270)
    return img


class train_DataSet(Dataset):
    def __init__(self, transform1, path, flag='train'):
        self.flag = flag
        self.transform1 = transform1
        self.haze_path, self.gt_path, self.t_path, self.A_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.haze_data_list.sort(key=lambda x: int(x[:5]))

        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:5]))

        self.t_data_list = os.listdir(self.t_path)

        self.length = len(self.haze_data_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        haze_name = self.haze_data_list[idx][:-4]
        num = haze_name.split('_')[0]
        A_name = haze_name.split('_')[1]

        haze_image = Image.open(self.haze_path + haze_name + '.bmp')
        gt_image = Image.open(self.gt_path + num + '.bmp')
        t_gth = Image.open(self.t_path + haze_name + '.bmp')
        A_gth = Image.open(self.A_path + A_name + '.bmp')

        # 数据增强
        if self.flag == 'train':
            [haze_image, gt_image, A_gth, t_gth] = data_aug([haze_image, gt_image, A_gth, t_gth])
            haze_image = np.asarray(haze_image)
            gt_image = np.asarray(gt_image)
            A_gth = np.asarray(A_gth)
            t_gth = np.asarray(t_gth)

        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)
            A_gth = self.transform1(A_gth)
            t_gth = self.transform1(t_gth)

        return haze_name, haze_image, gt_image, A_gth, t_gth


class test_DataSet(Dataset):
    def __init__(self, transform1, path):
        self.transform1 = transform1
        self.haze_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.haze_data_list.sort(key=lambda x: int(x[:-4]))

        self.length = len(self.haze_data_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        haze_name = self.haze_data_list[idx][:-4]

        haze_image = Image.open(self.haze_path + self.haze_data_list[idx])

        if self.transform1:
            haze_image = self.transform1(haze_image)

        return haze_name, haze_image
