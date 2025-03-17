import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.datasets import Test_ChangeDataset
from torchvision.transforms import Resize, CenterCrop, Normalize
import tqdm


def get_dataset_loaders(workers, batch_size=4):

    train_dataset = Test_ChangeDataset(
        os.path.join(data_dir, "Urban", "images_png_clip/"), os.path.join(data_dir, "Urban", "images_png_clip/"), os.path.join(data_dir, "Urban", "masks_png_clip/"), trainsize=256

    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    return train_loader




if __name__ == '__main__':



    Image.MAX_IMAGE_PIXELS = None

    # 创建数据集实例
    data_dir = r"D:\zly\2021LoveDA\Train/"
    transform = transforms.ToTensor()

    train_loader = get_dataset_loaders(5, batch_size=1)

    total_mean1 = torch.zeros(3)
    total_mean2 = torch.zeros(3)
    total_mean3 = torch.zeros(3)
    total_mean4 = torch.zeros(3)
    total_mean5 = torch.zeros(3)
    total_mean6 = torch.zeros(3)
    total_mean7 = torch.zeros(3)

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0



    target_labels = [1, 2, 3, 4, 5, 6, 7]
    total_means = [total_mean1, total_mean2, total_mean3, total_mean4, total_mean5, total_mean6, total_mean7]
    counts = [count1, count2, count3, count4, count5, count6, count7]

    for images_A, images_B, labels,_ in tqdm.tqdm(train_loader):

        labels = labels * 255
        labels = labels.squeeze(0)

        for idx, target_label in enumerate(target_labels):
            mask = torch.where(labels == target_label, torch.tensor(1), torch.tensor(0))
            filtered_images_A = images_A * mask.unsqueeze(1)


            non_zero_mask = filtered_images_A != 0
            num_non_zero = non_zero_mask.sum(dim=(0, 2, 3))

         
            if num_non_zero.sum() > 0:
                sum_non_zero = (filtered_images_A * non_zero_mask).sum(dim=(0, 2, 3))
                mean = sum_non_zero / num_non_zero

                total_means[idx] += mean
                counts[idx] += 1

    average_means = [total_means[i] / counts[i] if counts[i] > 0 else torch.zeros(3) for i in range(len(target_labels))]

    for idx, target_label in enumerate(target_labels):
        print(f"Average Mean for label {target_label}: {average_means[idx]}")

