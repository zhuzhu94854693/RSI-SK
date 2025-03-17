
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
from pathlib import Path
import collections

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from utils.datasets import Test_ChangeDataset, Test_ChangeDataset_nolabel

from torchvision.transforms import Resize, CenterCrop, Normalize

import datetime
import random
import os
import tqdm
import json
import argparse

from torch_poly_lr_decay import PolynomialLRDecay
import numpy as np
from itertools import chain
import itertools
import torch.nn.functional as F

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

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageEnhance

from network.class_pro import class_pro
import pickle
import torch.nn.functional as F

device = 'cuda'

path = r'/home/user/zly/data2/LEVIR_clip/sample/t0.10/'

seed = 45
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def get_dataset_loaders(workers, batch_size=4):

    train_dataset = Test_ChangeDataset(

        os.path.join(path, "train", "A/"), os.path.join(path, "train", "B/"), os.path.join(path, "train", "label/"), trainsize=256

    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)


    return train_loader




def histogram_matching(reference_tensor, target_tensor, hist_sum):
    def match_histograms(source, reference, hist_sum):
        matched = np.zeros_like(source, dtype=np.uint8)
        channels = ['R', 'G', 'B']
        for k, channel in enumerate(channels):
            source_hist, _ = np.histogram(source[:, :, k].ravel(), 256, [0, 256])
            reference_hist, _ = np.histogram(reference[:, :, k].ravel(), 256, [0, 256])

            cdf_source = source_hist.cumsum()
            cdf_source = 255 * cdf_source / cdf_source[-1]


            cdf_reference = reference_hist.cumsum()
            cdf_reference = 255 * cdf_reference / cdf_reference[-1]




            cdf_hist = hist_sum[channel]['image_diff'].cumsum()
            cdf_hist = cdf_hist / cdf_hist[-1]





            lookup_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):


                idx = np.argmin(np.abs(cdf_reference - cdf_source[i]))
                lookup_table[i] = idx if cdf_reference[idx] <= cdf_source[i] else max(0, idx - 1)

            indices = source[:, :, k].astype(np.int64)
            matched[:, :, k] = np.take(lookup_table, indices)

        return matched

    matched_tensors = []
    for i in range(reference_tensor.shape[0]):
        reference_image = (reference_tensor[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        target_image = (target_tensor[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        matched_image_np = match_histograms(target_image, reference_image, hist_sum)
        matched_image = torch.from_numpy(matched_image_np.transpose(2, 0, 1).astype(np.float32) / 255).to(target_tensor.device)

        matched_tensors.append(matched_image)

    return torch.stack(matched_tensors)




def train(labeled_loader,  num_classes, device, ):
    global dif_hist1, dif_aug_hist1, dif_image1



    target_labels = [0, 1, 2, 3, 4, 5, 6]
    total_means_images1 = [torch.zeros(3).to(device) for _ in target_labels]
    total_means_images2 = [torch.zeros(3).to(device) for _ in target_labels]

    counts_images1 = [0] * len(target_labels)
    counts_images2 = [0] * len(target_labels)


    labeled_loader_iter = iter(labeled_loader)

    total_steps = len(labeled_loader)

    with tqdm.tqdm(total=total_steps, desc="Training with labeled and unlabeled data") as pbar:
        for idx in range(total_steps):

            if idx < len(labeled_loader):
                labeled_batch = next(labeled_loader_iter)
                images1_labeled, images2_labeled, masks,_ = labeled_batch
                images1_labeled = images1_labeled.to(device)
                images2_labeled = images2_labeled.to(device)
                masks = masks.to(device)


                mask_all_zero_indices = (masks == 0).all(dim=1).all(dim=1).all(dim=1)

                if mask_all_zero_indices.any():
                    images1_all_zero = images1_labeled[mask_all_zero_indices]
                    images2_all_zero = images2_labeled[mask_all_zero_indices]


                    classA, classB, classproA, classproB = class_pro(images1_all_zero, images2_all_zero)

                    for idx, target_label in enumerate(target_labels):
                        mask = torch.where(classA == target_label, torch.tensor(1).to(device),
                                           torch.tensor(0).to(device))

                        # 处理 images1_all_zero
                        filtered_images1 = images1_all_zero * mask.unsqueeze(1)
                        non_zero_mask1 = filtered_images1 != 0
                        num_non_zero1 = non_zero_mask1.sum(dim=(0, 2, 3))

                        if num_non_zero1.sum() > 0:
                            sum_non_zero1 = (filtered_images1 * non_zero_mask1).sum(dim=(0, 2, 3))
                            mean1 = sum_non_zero1 / num_non_zero1
                            total_means_images1[idx] += mean1
                            counts_images1[idx] += 1

                        # 处理 images2_all_zero
                        filtered_images2 = images2_all_zero * mask.unsqueeze(1)
                        non_zero_mask2 = filtered_images2 != 0
                        num_non_zero2 = non_zero_mask2.sum(dim=(0, 2, 3))

                        if num_non_zero2.sum() > 0:
                            sum_non_zero2 = (filtered_images2 * non_zero_mask2).sum(dim=(0, 2, 3))
                            mean2 = sum_non_zero2 / num_non_zero2
                            total_means_images2[idx] += mean2
                            counts_images2[idx] += 1



            pbar.update(1)


    average_means_images1 = [
        total_means_images1[i] / counts_images1[i] if counts_images1[i] > 0 else torch.zeros(3).to(device)
        for i in range(len(target_labels))]
    average_means_images2 = [
        total_means_images2[i] / counts_images2[i] if counts_images2[i] > 0 else torch.zeros(3).to(device)
        for i in range(len(target_labels))]

    for idx, target_label in enumerate(target_labels):
        print(f"Average Mean for label {target_label} in images1_all_zero: {average_means_images1[idx]}")
        print(f"Average Mean for label {target_label} in images2_all_zero: {average_means_images2[idx]}")
      


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', nargs='?', type=int, default=0,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--swa_start', nargs='?', type=int, default=1)
    parser.add_argument('--lr', nargs='?', type=float, default=5e-3,
                        help='Learning Rate')
    parser.add_argument('--model', nargs='?', type=str, default='Unet')
    parser.add_argument('--swa', nargs='?', type=bool, default=True)

    parser.add_argument('--start_epoch', default=1, type=int)

    parser.add_argument('--r', dest='resume', default=False, type=bool)

    arg = parser.parse_args()

    semisemi = 10
    num_classes = 2
    model_name = arg.model
    print(model_name)
    learning_rate = arg.lr
    num_epochs = arg.n_epoch
    batch_size = arg.batch_size

    history = collections.defaultdict(list)

    train_loader = get_dataset_loaders(5, batch_size)


    train_hist = train(train_loader, num_classes, device)




