from PIL import Image
import cv2
import numpy as np

from pathlib import Path
import collections
from ranger import Ranger
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from utils.datasets import ATDataset
from utils.loss import CrossEntropyLoss2d, mIoULoss2d, FocalLoss2d, LovaszLoss2d
from torchvision.transforms import Resize, CenterCrop, Normalize
from utils.metrics import Metrics
# from utils.lr_scheduler import LR_Scheduler
# from models.atcdnet.atcdnet import ATCDNet
import datetime
import random
import os
import tqdm
import json
import argparse
from logsetting import get_log
from torch_poly_lr_decay import PolynomialLRDecay

device = 'cuda'
path = r'/home/user/zly/data2/WHU_clip/sample/orgion/'


seed = 42  # 可以选择任意整数作为种子
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_dataset_loaders(workers, batch_size=1):
    target_size = 256

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    trainval_transform = transforms.Compose(
        [
            #             JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            # JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            # JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90),
            # transforms.RandomRotation(180),
            # transforms.RandomRotation(270),
            transforms.ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    target_transform = transforms.Compose(
        [
            # JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            # JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            # JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90),
            # transforms.RandomRotation(180),
            # transforms.RandomRotation(270),
            transforms.ToTensor(),
            # Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            Normalize(mean=mean, std=std)
        ]
    )

    test_dataset = ATDataset(
        os.path.join(path, "test", "A"), os.path.join(path, "test", "B"), os.path.join(path, "test", "label"),
        trainval_transform, test_transform, target_transform

    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    return test_loader


def test(loader, num_classes, device, net):
    #num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    net.eval()

    result_folder = r'/home/user/zly/data2/WHU_clip/result/MSAANet/t0.10'

    # 创建保存结果的文件夹
    os.makedirs(result_folder, exist_ok=True)

    for i, (images1, images2, masks) in enumerate(loader):
        images1 = images1.to(device)
        images2 = images2.to(device)
        masks = masks.to(device)

        assert images1.size()[2:] == images2.size()[2:] == masks.size()[
                                                           2:], "resolutions for images and masks are in sync"

        # num_samples += int(images1.size(0))

        outputs = net(images1, images2)

        assert outputs.size()[2:] == masks.size()[2:], "resolutions for predictions and masks are in sync"
        assert outputs.size()[1] == num_classes, "classes for predictions and dataset are in sync"

        # loss = criterion(outputs, masks.float())  ##BCELoss
        # loss = criterion(outputs, masks.long())
        # running_loss += loss.item()
        outputs = outputs[:, 0, :, :]  # .unsqueeze(1)
        masks = masks[:, 0, :, :]  # .unsqueeze(1)

        for mask, output in zip(masks, outputs):
            metrics.add(mask, output)

            # 将预测结果保存为 PNG 文件
            # # output = torch.argmax(output, dim=0).cpu().numpy().astype(np.uint8) * 255
            # output = output.cpu().detach().numpy().astype(np.uint8) * 255
           # output_image = Image.fromarray(output)
            threshold = 0.5
            output = (output > threshold).float()

            output = output.cpu().detach().numpy() * 255
            output = output.astype(np.uint8)
            output_image = Image.fromarray(output)  # 仅使用第一个通道作为灰度图像

            image_path = os.path.basename(loader.dataset.image_path1[i])
            output_image.save(os.path.join(result_folder, f"{image_path}.tif"))


    # assert num_samples > 0, "dataset contains validation images and labels"

    return {
        # "loss": running_loss / num_samples,
        "precision": metrics.get_precision(),
        "recall": metrics.get_recall(),
        "f_score": metrics.get_f_score(),
        "iou": metrics.get_fg_iou(),
        "oa": metrics.get_oa()
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')

    arg = parser.parse_args()

    num_classes = 2

    batch_size = arg.batch_size

    history = collections.defaultdict(list)

    net = torch.load(r"/home/user/zly/data2/WHU_clip/pth/MSAANet/t0.10/model_best.pth")
    # net.load_state_dict(state)
    if torch.cuda.device_count() > 1:
        print("using multi gpu")
        net = torch.nn.DataParallel(net, device_ids=[0, 1])
    else:
        print('using one gpu')

    test_loader = get_dataset_loaders(5, batch_size)

    today = str(datetime.date.today())
    logger = get_log('/home/user/zly/data2/WHU_clip/result/MSAANet/MSAANet_t0.10_test.txt')

    test_hist = test(test_loader, num_classes, device, net)
    logger.info((  # 'loss={}'.format(val_hist["loss"]),
        'precision={}'.format(test_hist["precision"]),
        'recall={}'.format(test_hist["recall"]),
        'f_score={}'.format(test_hist["f_score"]),
        'iou={}'.format(test_hist["iou"]),
        'oa={}'.format(test_hist["oa"])))

    for k, v in test_hist.items():
        history["test " + k].append(v)
