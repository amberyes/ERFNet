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
    def __init__(self, transform1, path=None):
        self.transform1 = transform1
        self.haze_path, self.gt_path, self.re_path = path

        self.haze_data_list = os.listdir(self.haze_path)
        self.haze_data_list.sort(key=lambda x: int(x[:-4]))

        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))

        self.re_data_list = os.listdir(self.re_path)
        self.re_data_list.sort(key=lambda x: int(x[:5]))

        self.length = len(os.listdir(self.haze_path))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        haze_name = self.haze_data_list[idx][:-4]
        num = haze_name.split('_')[0]

        haze_image = Image.open(self.haze_path + haze_name + '.bmp')
        gt_image = Image.open(self.gt_path + num + '.bmp')
        re_image = Image.open(self.re_path + haze_name + '.bmp')

        # 数据增强

        [haze_image, gt_image, re_image] = data_aug([haze_image, gt_image, re_image])
        haze_image = np.asarray(haze_image)
        gt_image = np.asarray(gt_image)
        re_image = np.asarray(re_image)

        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)
            re_image = self.transform1(re_image)

        return haze_name, haze_image, gt_image, re_image


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


class Ntire_DataSet(Dataset):
    def __init__(self, transform1, path=None):
        self.transform1 = transform1
        self.haze_path, self.gt_path = path

        self.haze_data_list = os.listdir(self.haze_path)
        self.haze_data_list.sort(key=lambda x: int(x[:-4]))

        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))

        self.length = len(os.listdir(self.haze_path))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        haze_name = self.haze_data_list[idx][:-4]

        haze_image = Image.open(self.haze_path + self.haze_data_list[idx])
        gt_image = Image.open(self.gt_path + self.haze_data_list[idx])

        # 数据增强

        [haze_image, gt_image] = data_aug([haze_image, gt_image])
        haze_image = np.asarray(haze_image)
        gt_image = np.asarray(gt_image)

        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)

        return haze_name, haze_image, gt_image
