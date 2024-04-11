import os
from glob import glob
from math import exp

import torch
import torch.utils.data as data
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DatasetFromFolder(data.Dataset):
    def __init__(self,  other_list, MRI_list=None, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.other_list = other_list
        self.MRI_list = MRI_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        other_img = cv2.imread(self.other_list[index],cv2.IMREAD_GRAYSCALE)
        other_img = self.transform(other_img)
        if self.MRI_list is not None:
            MRI_img = cv2.imread(self.MRI_list[index],cv2.IMREAD_GRAYSCALE)
            MRI_img = self.transform(MRI_img)
            other_MRI_img = torch.cat((other_img, MRI_img), -3)
            return other_img, MRI_img, other_MRI_img
        return other_img

    def __len__(self):
        return len(self.other_list)

import pywt
import numpy as np

def entropy(X):
    X = X.flatten()
    X = np.uint8(X)
    n = len(X)
    counts = np.bincount(X)
    probs = counts[np.nonzero(counts)]/n
    en = 0
    for i in range(len(probs)):
        en = en - probs[i] * np.log(probs[i]/np.log(2))
    return en

def haar_weight(MRI, OTHER):
    cA, (cH, cV, cD) = pywt.dwt2(MRI, 'haar')
    MRI_SUM = (entropy(cH) + entropy(cV) + entropy(cD)) / 3.0
    cA, (cH, cV, cD) = pywt.dwt2(OTHER, 'haar')
    # OTHERI_SUM = abs(np.average(cH)) + abs(np.average(cV)) + abs(np.average(cD))
    # OTHERI_SUM = (cH.std() + cV.std() + cD.std()) / 3
    OTHERI_SUM = (entropy(cH) + entropy(cV) + entropy(cD))/3.0

    MRI_SUM, OTHERI_SUM = MRI_SUM/(OTHERI_SUM+MRI_SUM)/3, OTHERI_SUM/(OTHERI_SUM+MRI_SUM)/3

    MRI_weight = exp(MRI_SUM) / (exp(MRI_SUM) + exp(OTHERI_SUM))
    OTHERI_weight = exp(OTHERI_SUM) / (exp(MRI_SUM) + exp(OTHERI_SUM))

    return MRI_weight, OTHERI_weight


class DatasetFromFolder_w(data.Dataset):
    def __init__(self,  other_list, MRI_list=None, transform=None):
        super(DatasetFromFolder_w, self).__init__()
        self.other_list = other_list
        self.MRI_list = MRI_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        other_img = cv2.imread(self.other_list[index],cv2.IMREAD_GRAYSCALE)
        if self.MRI_list is not None:
            MRI_img = cv2.imread(self.MRI_list[index],cv2.IMREAD_GRAYSCALE)
            MRI_weight, other_weight = haar_weight(MRI_img, other_img)
            MRI_weight = torch.FloatTensor(np.asarray(MRI_weight).reshape([-1]))
            other_weight = torch.FloatTensor(np.asarray(other_weight).reshape([-1]))
            other_img = self.transform(other_img)
            MRI_img = self.transform(MRI_img)
            other_MRI_img = torch.cat((other_img, MRI_img), -3)
            return other_img, MRI_img, other_MRI_img, other_weight, MRI_weight
        other_img = self.transform(other_img)
        return other_img

    def __len__(self):
        return len(self.other_list)



if __name__ == '__main__':
    MRI_PATH = 'E:/python-project/pytorch17/Medical_image_fusion/data/data1/train/pair1/'
    OTHER_PATH = 'E:/python-project/pytorch17/Medical_image_fusion/data/data1/train/pair1/'
    MRI_list = glob(MRI_PATH + '/*.jpg')
    OTHER_list = glob(OTHER_PATH + '/*.jpg')
    # 顺序读取
    train_set = DatasetFromFolder(MRI_list + OTHER_list)
    print(len(train_set))
    # 成对读取
    train_set_2 = DatasetFromFolder(MRI_list, OTHER_list)
    print(len(train_set_2))

    train_loader = DataLoader(train_set_2, shuffle=True, batch_size=8)
    for i,(other_img, MRI_img, other_MRI_img) in enumerate(train_loader):
        print(other_img.shape)  # torch.Size([8, 1, 256, 256])
        print(MRI_img.shape)  # torch.Size([8, 1, 256, 256])
        print(other_MRI_img.shape)  # torch.Size([8, 2, 256, 256])
        break