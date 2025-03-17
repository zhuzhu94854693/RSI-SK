import torch
import torch.nn as nn
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)


def filter_layers(layer, layer_class):
  
    filtered = torch.zeros_like(layer)


    for c in range(layer.shape[1]):  

        max_val = torch.max(layer[:, c, :, :]).item()
        min_val = torch.min(layer[:, c, :, :]).item()
        threshold = (max_val - min_val)

        max_val_class = torch.max(layer_class[:, c, :, :]).item()
        min_val_class = torch.min(layer_class[:, c, :, :]).item()
        threshold_class = (max_val_class - min_val_class)

        mask1 = (layer[:, c, :, :] >= threshold * 0.8 + min_val) & (layer_class[:, c, :, :] >= threshold_class * 0.8 + min_val_class)
        mask2 = (layer[:, c, :, :] < threshold * 0.2 + min_val) & (layer_class[:, c, :, :] < threshold_class * 0.2 + min_val_class) 

        filtered[:, c, :, :] = torch.where(mask1, layer[:, c, :, :] + layer_class[:, c, :, :], torch.zeros_like(layer[:, c, :, :]))
        filtered[:, c, :, :] = torch.where(mask2, torch.abs(layer[:, c, :, :] - layer_class[:, c, :, :]), filtered[:, c, :, :])

    return filtered


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ClassAttention(nn.Module):
    def __init__(self, in_channels, class_channels, ratio=16):
        super(ClassAttention, self).__init__()
        self.fc1 = nn.Conv2d(class_channels, class_channels // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(class_channels // ratio, in_channels, kernel_size=1, bias=False)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, class_input):
        avg_out = torch.mean(class_input, dim=[2, 3], keepdim=True)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        return self.sigmoid(avg_out) * x



class Unet(nn.Module):
    def __init__(self, channel=32):
        super(Unet, self).__init__()


        vgg16_bn_1 = models.vgg16_bn(pretrained=True)
        self.inc_A_B = vgg16_bn_1.features[:5]
        self.down1 = vgg16_bn_1.features[5:12]  # 128
        self.down2 = vgg16_bn_1.features[12:22]  # 256
        self.down3 = vgg16_bn_1.features[22:32]  # 512
        self.down4 = vgg16_bn_1.features[32:42]  # 512


        vgg16_bn_class = models.vgg16_bn(pretrained=True)

        self.inc_classA_B = nn.Sequential(nn.Conv2d(7, 64, kernel_size=3, padding=1),  # 7 input channels for classA and classB
            *vgg16_bn_class.features[1:5]
        )


        self.down1_class = vgg16_bn_class.features[5:12]
        self.down2_class = vgg16_bn_class.features[12:22]
        self.down3_class = vgg16_bn_class.features[22:32]
        self.down4_class = vgg16_bn_class.features[32:42]


        self.class_A_attention_1 = ClassAttention(64, 64)
        self.class_A_attention_2 = ClassAttention(128, 128)
        self.class_A_attention_3 = ClassAttention(256, 256)
        self.class_A_attention_4 = ClassAttention(512, 512)
        self.class_A_attention_5 = ClassAttention(512, 512)

        self.class_B_attention_1 = ClassAttention(64, 64)
        self.class_B_attention_2 = ClassAttention(128, 128)
        self.class_B_attention_3 = ClassAttention(256, 256)
        self.class_B_attention_4 = ClassAttention(512, 512)
        self.class_B_attention_5 = ClassAttention(512, 512)


        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)



        self.atten_A_channel_1=ChannelAttention(64)
        self.atten_A_channel_2=ChannelAttention(128)
        self.atten_A_channel_3=ChannelAttention(256)
        self.atten_A_channel_4=ChannelAttention(512)
        self.atten_A_channel_5=ChannelAttention(512)

        self.atten_A_spatial_1=SpatialAttention()
        self.atten_A_spatial_2=SpatialAttention()
        self.atten_A_spatial_3=SpatialAttention()
        self.atten_A_spatial_4=SpatialAttention()
        self.atten_A_spatial_5=SpatialAttention()

        self.atten_B_channel_1=ChannelAttention(64)
        self.atten_B_channel_2=ChannelAttention(128)
        self.atten_B_channel_3=ChannelAttention(256)
        self.atten_B_channel_4=ChannelAttention(512)
        self.atten_B_channel_5=ChannelAttention(512)

        self.atten_B_spatial_1=SpatialAttention()
        self.atten_B_spatial_2=SpatialAttention()
        self.atten_B_spatial_3=SpatialAttention()
        self.atten_B_spatial_4=SpatialAttention()
        self.atten_B_spatial_5=SpatialAttention()



        self.up4 = UpConv(512, 512)
        self.upconv4 = DoubleConv(1024, 512)

        self.up3 = UpConv(512, 256)
        self.upconv3 = DoubleConv(512, 256)

        self.up2 = UpConv(256, 128)
        self.upconv2 = DoubleConv(256, 128)

        self.up1 = UpConv(128, 64)
        self.upconv1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)



    def forward(self, A, B, classA, classB, classC, classD, classE, preduA, use_ema, with_label): #

        layer1_A = (self.inc_A_B(A))
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)
        layer4_A = self.down3(layer3_A)
        layer5_A = self.down4(layer4_A)

        layer1_B = (self.inc_A_B(B))
        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)
        layer4_B = self.down3(layer3_B)
        layer5_B = self.down4(layer4_B)


        layer1_A = layer1_A.mul(self.atten_A_channel_1(layer1_A))
        layer1_A = layer1_A.mul(self.atten_A_spatial_1(layer1_A))



        layer1_B = layer1_B.mul(self.atten_B_channel_1(layer1_B))
        layer1_B = layer1_B.mul(self.atten_B_spatial_1(layer1_B))




        layer2_A = layer2_A.mul(self.atten_A_channel_2(layer2_A))
        layer2_A = layer2_A.mul(self.atten_A_spatial_2(layer2_A))


        layer2_B = layer2_B.mul(self.atten_B_channel_2(layer2_B))
        layer2_B = layer2_B.mul(self.atten_B_spatial_2(layer2_B))


        layer3_A = layer3_A.mul(self.atten_A_channel_3(layer3_A))
        layer3_A = layer3_A.mul(self.atten_A_spatial_3(layer3_A))


        layer3_B = layer3_B.mul(self.atten_B_channel_3(layer3_B))
        layer3_B = layer3_B.mul(self.atten_B_spatial_3(layer3_B))


        layer4_A = layer4_A.mul(self.atten_A_channel_4(layer4_A))
        layer4_A = layer4_A.mul(self.atten_A_spatial_4(layer4_A))


        layer4_B = layer4_B.mul(self.atten_B_channel_4(layer4_B))
        layer4_B = layer4_B.mul(self.atten_B_spatial_4(layer4_B))


        layer5_A = layer5_A.mul(self.atten_A_channel_5(layer5_A))
        layer5_A = layer5_A.mul(self.atten_A_spatial_5(layer5_A))


        layer5_B = layer5_B.mul(self.atten_B_channel_5(layer5_B))
        layer5_B = layer5_B.mul(self.atten_B_spatial_5(layer5_B))



        layer1_classA = self.inc_classA_B(classA)
        layer2_classA = self.down1_class(layer1_classA)
        layer3_classA = self.down2_class(layer2_classA)
        layer4_classA = self.down3_class(layer3_classA)
        layer5_classA = self.down4_class(layer4_classA)

        layer1_classB = self.inc_classA_B(classB)
        layer2_classB = self.down1_class(layer1_classB)
        layer3_classB = self.down2_class(layer2_classB)
        layer4_classB = self.down3_class(layer3_classB)
        layer5_classB = self.down4_class(layer4_classB)

        layer1_classE = self.inc_classA_B(classE)


        layer1_classF = self.inc_classA_B(preduA)

        layer1_classA = layer1_classA.mul(self.atten_A_channel_1(layer1_classA))
        layer1_classA = layer1_classA.mul(self.atten_A_spatial_1(layer1_classA))


        layer1_classB = layer1_classB.mul(self.atten_B_channel_1(layer1_classB))
        layer1_classB = layer1_classB.mul(self.atten_B_spatial_1(layer1_classB))


        layer1_classE = layer1_classE.mul(self.atten_A_channel_1(layer1_classE))
        layer1_classE = layer1_classE.mul(self.atten_A_spatial_1(layer1_classE))


        layer1_classF = layer1_classF.mul(self.atten_B_channel_1(layer1_classF))
        layer1_classF = layer1_classF.mul(self.atten_B_spatial_1(layer1_classF))


        layer2_classA = layer2_classA.mul(self.atten_A_channel_2(layer2_classA))
        layer2_classA = layer2_classA.mul(self.atten_A_spatial_2(layer2_classA))



        layer2_classB = layer2_classB.mul(self.atten_B_channel_2(layer2_classB))
        layer2_classB = layer2_classB.mul(self.atten_B_spatial_2(layer2_classB))


        layer3_classA = layer3_classA.mul(self.atten_A_channel_3(layer3_classA))
        layer3_classA = layer3_classA.mul(self.atten_A_spatial_3(layer3_classA))


        layer3_classB = layer3_classB.mul(self.atten_B_channel_3(layer3_classB))
        layer3_classB = layer3_classB.mul(self.atten_B_spatial_3(layer3_classB))


        layer4_classA = layer4_classA.mul(self.atten_A_channel_4(layer4_classA))
        layer4_classA = layer4_classA.mul(self.atten_A_spatial_4(layer4_classA))


        layer4_classB = layer4_classB.mul(self.atten_B_channel_4(layer4_classB))
        layer4_classB = layer4_classB.mul(self.atten_B_spatial_4(layer4_classB))


        layer5_classA = layer5_classA.mul(self.atten_A_channel_5(layer5_classA))
        layer5_classA = layer5_classA.mul(self.atten_A_spatial_5(layer5_classA))


        layer5_classB = layer5_classB.mul(self.atten_B_channel_5(layer5_classB))
        layer5_classB = layer5_classB.mul(self.atten_B_spatial_5(layer5_classB))


        layer1 = torch.abs(layer1_B - layer1_A)
        layer2 = torch.abs(layer2_B - layer2_A)
        layer3 = torch.abs(layer3_B - layer3_A)
        layer4 = torch.abs(layer4_B - layer4_A)
        layer5 = torch.abs(layer5_B - layer5_A)

        layer1_class = torch.abs(layer1_classB - layer1_classA)
        layer2_class = torch.abs(layer2_classB - layer2_classA)
        layer3_class = torch.abs(layer3_classB - layer3_classA)
        layer4_class = torch.abs(layer4_classB - layer4_classA)
        layer5_class = torch.abs(layer5_classB - layer5_classA)



        layer1_class = layer1 + layer1_class
        layer2_class = layer2 + layer2_class
        layer3_class = layer3 + layer3_class
        layer4_class = layer4 + layer4_class
        layer5_class = layer5 + layer5_class




        x = self.up4(layer5)
        x = torch.cat([x, layer4], dim=1)
        x = self.upconv4(x)

        x = self.up3(x)
        x = torch.cat([x, layer3], dim=1)
        x = self.upconv3(x)

        x = self.up2(x)
        x = torch.cat([x, layer2], dim=1)
        x = self.upconv2(x)

        x = self.up1(x)
        x = torch.cat([x, layer1], dim=1)
        x = self.upconv1(x)

        x = self.final_conv(x)

        x_class = self.up4(layer5_class)
        x_class = torch.cat([x_class, layer4_class], dim=1)
        x_class = self.upconv4(x_class)

        x_class = self.up3(x_class)
        x_class = torch.cat([x_class, layer3_class], dim=1)
        x_class = self.upconv3(x_class)

        x_class = self.up2(x_class)
        x_class = torch.cat([x_class, layer2_class], dim=1)
        x_class = self.upconv2(x_class)

        x_class = self.up1(x_class)
        x_class = torch.cat([x_class, layer1_class], dim=1)
        x_class = self.upconv1(x_class)

        x_class = self.final_conv(x_class)

        if use_ema is True and torch.sum(~with_label) > 0:
            x_s = filter_layers(x, x_class)
        else:
            x_s =x


        loss_laycha_1 = F.mse_loss(layer1_classA, layer1_classE)

        loss_t = loss_laycha_1

        loss_laycha_1 = F.mse_loss(layer1_classB, layer1_classF)

        loss_t2 = loss_laycha_1

        return x, x_class, x_s, loss_t, loss_t2








