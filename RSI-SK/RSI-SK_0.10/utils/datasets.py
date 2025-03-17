
from pathlib import Path
import torch
import matplotlib.image as mping
import glob
from PIL import Image
import torch.utils.data
from torchvision.transforms import Compose, Normalize
from skimage import io
from torchvision.transforms import transforms
import numpy as np
import os


import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageEnhance




class Test_ChangeDataset(data.Dataset):
    def __init__(self, inputs1, inputs2, target, trainsize):
        self.trainsize = trainsize

        image_root_A =  inputs1
        image_root_B =  inputs2
        gt_root = target
        self.images_A = [image_root_A + f for f in os.listdir(image_root_A) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_B = [image_root_B + f for f in os.listdir(image_root_B) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images_A = sorted(self.images_A)
        self.images_B = sorted(self.images_B)
        self.gts = sorted(self.gts)
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.size = len(self.images_A)

    def __getitem__(self, index):

        image_A = self.rgb_loader(self.images_A[index])
        image_B = self.rgb_loader(self.images_B[index])
        gt = self.binary_loader(self.gts[index])


        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)
        gt = self.gt_transform(gt)
        file_name = self.images_A[index].split('/')[-1][:-len(".png")]

        return image_A, image_B, gt, file_name

    def filter_files(self):
        assert len(self.images_A) == len(self.gts)
        assert len(self.images_A) == len(self.images_B)
        images_A = []
        images_B = []
        gts = []
        for img_A_path, img_B_path, gt_path in zip(self.images_A, self.images_B, self.gts):
            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)
            gt = Image.open(gt_path)
            if img_A.size == img_B.size:
                if img_A.size == gt.size:
                    images_A.append(img_A_path)
                    images_B.append(img_B_path)
                    gts.append(gt_path)

        self.images_A = images_A
        self.images_B = images_B
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size




class Test_ChangeDataset_nolabel(data.Dataset):
    def __init__(self, inputs1, inputs2, trainsize):
        self.trainsize = trainsize
        # get filenames
        image_root_A =  inputs1
        image_root_B =  inputs2

        self.images_A = [image_root_A + f for f in os.listdir(image_root_A) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_B = [image_root_B + f for f in os.listdir(image_root_B) if f.endswith('.jpg') or f.endswith('.png')]

        self.images_A = sorted(self.images_A)
        self.images_B = sorted(self.images_B)

        self.filter_files()

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.size = len(self.images_A)

    def __getitem__(self, index):

        image_A = self.rgb_loader(self.images_A[index])
        image_B = self.rgb_loader(self.images_B[index])


        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)

        file_name = self.images_A[index].split('/')[-1][:-len(".png")]

        return image_A, image_B, file_name

    def filter_files(self):

        images_A = []
        images_B = []

        for img_A_path, img_B_path in zip(self.images_A, self.images_B):
            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)

            if img_A.size == img_B.size:

                images_A.append(img_A_path)
                images_B.append(img_B_path)


        self.images_A = images_A
        self.images_B = images_B


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

