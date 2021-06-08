# -*- coding: utf-8 -*-
# @Time : 2021/6/2 20:14
# @Author : xmhh
# @File : TrashDataloader.py
# @Software: PyCharm
import os
from glob import glob

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import torch.utils.data as data
import numpy as np

train_transform = A.Compose([
    # reszie
    A.PadIfNeeded(min_height=600, min_width=600, value=0),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.8),
        A.Transpose(p=0.5),
    ]),
    A.RandomGamma(gamma_limit=(85, 115), p=1),
    A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3)),
    A.GaussNoise(var_limit=15,p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=600, min_width=600, value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class TrashFolder(data.Dataset):

    def __init__(self, imgs_dir, transforms=None):
        self.transforms = transforms
        self.imgs_dir = imgs_dir
        self.idx_and_img = {}
        num = 0
        for paths in os.listdir(imgs_dir):
            for img in os.listdir(os.path.join(imgs_dir, paths)):
                self.idx_and_img[num] = (img, paths)
                num += 1


        self.transforms = transforms

    def __getitem__(self, index):
        img_name, label = self.idx_and_img[index]
        img_file = os.path.join(self.imgs_dir, str(label), img_name)
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        # cv2.imshow("", image)
        transformed = self.transforms(image=image)
        img = transformed['image']

        return img, int(label)

    def __len__(self):
        return len(self.idx_and_img)

if __name__ == '__main__':
    data_dir = r"F:\dosth\trash_trainval\trainval\trash_trainval"
    train_imgs_dir = os.path.join(data_dir, "train")
    val_imgs_dir = os.path.join(data_dir, "val")


    train_data = TrashFolder(train_imgs_dir, transforms=train_transform)
    for tr_idx, (train_data_img, label) in enumerate(train_data):
        print(train_data_img.shape)
        print(label)
        cv2.imshow('data', train_data_img)
        cv2.waitKey(0)