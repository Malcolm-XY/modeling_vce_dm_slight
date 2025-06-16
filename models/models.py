# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:06:57 2024

@author: 18307
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# %% mscnn
class MSCNN_2_2layers_adaptive_avgpool_3(nn.Module):
    """
    三分支网络，每个分支结构与原网络类似，最终输出拼接后分类。
    """
    def __init__(self, channels=3, num_classes=3):
        super(MSCNN_2_2layers_adaptive_avgpool_3, self).__init__()

        # 分支1
        self.branch1_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.branch1_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.AdaptiveMaxPool2d((1, 1))

        # 分支2
        self.branch2_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.branch2_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.AdaptiveMaxPool2d((1, 1))

        # 全连接层，用于整合三个分支的输出
        self.fc1 = nn.Linear(in_features=64 * 2, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        # 分支1
        x1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        x1 = self.branch1_pool1(x1)
        x1 = F.relu(self.branch1_bn2(self.branch1_conv2(x1)))
        x1 = self.branch1_pool2(x1)  # 输出 (batch_size, 64, 1, 1)

        # 分支2
        x2 = F.relu(self.branch2_bn1(self.branch2_conv1(x)))
        x2 = self.branch2_pool1(x2)
        x2 = F.relu(self.branch2_bn2(self.branch2_conv2(x2)))
        x2 = self.branch2_pool2(x2)  # 输出 (batch_size, 64, 1, 1)

        # 拼接两个分支的输出
        x_concat = torch.cat((x1, x2), dim=1)

        # 展平层
        x = x_concat.view(x_concat.size(0), -1)
        x = F.relu(self.fc1(x)) 
        
        # 最终分类        
        output = self.fc2(x)  # 输入到 fc2
        return output

class MSCNN_3_2layers_cv_235_adaptive_maxpool_3(nn.Module):
    """
    三分支网络，每个分支结构与原网络类似，最终输出拼接后分类。
    """
    def __init__(self, channels=3, num_classes=3):
        super(MSCNN_3_2layers_cv_235_adaptive_maxpool_3, self).__init__()

        # 分支1
        self.branch1_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.branch1_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.AdaptiveMaxPool2d((1, 1))

        # 分支2
        self.branch2_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.branch2_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.AdaptiveMaxPool2d((1, 1))

        # 分支3
        self.branch3_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.branch3_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.AdaptiveMaxPool2d((1, 1))

        # 全连接层，用于整合三个分支的输出
        self.fc1 = nn.Linear(in_features=64 * 3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # 分支1
        x1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        x1 = self.branch1_pool1(x1)
        x1 = F.relu(self.branch1_bn2(self.branch1_conv2(x1)))
        x1 = self.branch1_pool2(x1)  # 输出 (batch_size, 64, 1, 1)

        # 分支2
        x2 = F.relu(self.branch2_bn1(self.branch2_conv1(x)))
        x2 = self.branch2_pool1(x2)
        x2 = F.relu(self.branch2_bn2(self.branch2_conv2(x2)))
        x2 = self.branch2_pool2(x2)  # 输出 (batch_size, 64, 1, 1)

        # 分支3
        x3 = F.relu(self.branch3_bn1(self.branch3_conv1(x)))
        x3 = self.branch3_pool1(x3)
        x3 = F.relu(self.branch3_bn2(self.branch3_conv2(x3)))
        x3 = self.branch3_pool2(x3)  # 输出 (batch_size, 64, 1, 1)

        # 拼接三个分支的输出
        x_concat = torch.cat((x1, x2, x3), dim=1)  # 输出 (batch_size, 192, 1, 1)

        # 展平层
        x = x_concat.view(x_concat.size(0), -1)  # 展平为 (batch_size, 192)
        x = F.relu(self.fc1(x))  # 输入到 fc1，确保 fc1 的 in_features 为 192
        
        # 最终分类        
        output = self.fc2(x)  # 输入到 fc2
        return output

class MSCNN_2layers_adaptive_avgpool_2(nn.Module):
    """
    三分支网络，每个分支结构与原网络类似，最终输出拼接后分类。
    """
    def __init__(self, channels=3, num_classes=3):
        super(MSCNN_2layers_adaptive_avgpool_2, self).__init__()

        # 分支1
        self.branch1_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.branch1_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.AdaptiveMaxPool2d((1, 1))
        #
        self.branch1_fc = nn.Linear(in_features=64, out_features=32)

        # 分支2
        self.branch2_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.branch2_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.AdaptiveMaxPool2d((1, 1))
        #
        self.branch2_fc = nn.Linear(in_features=64, out_features=32)

        # 分支3
        self.branch3_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.branch3_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=1)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.AdaptiveMaxPool2d((1, 1))
        #
        self.branch3_fc = nn.Linear(in_features=64, out_features=32)

        # 全连接层，用于整合三个分支的输出
        self.fc1 = nn.Linear(in_features=64 * 3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
        self.final_fc = nn.Linear(in_features=32 * 3, out_features=num_classes)

    def forward(self, x):
        # 分支1
        x1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        x1 = self.branch1_pool1(x1)
        x1 = F.relu(self.branch1_bn2(self.branch1_conv2(x1)))
        x1 = self.branch1_pool2(x1)

        # 分支2
        x2 = F.relu(self.branch2_bn1(self.branch2_conv1(x)))
        x2 = self.branch2_pool1(x2)
        x2 = F.relu(self.branch2_bn2(self.branch2_conv2(x2)))
        x2 = self.branch2_pool2(x2)

        # 分支3
        x3 = F.relu(self.branch3_bn1(self.branch3_conv1(x)))
        x3 = self.branch3_pool1(x3)
        x3 = F.relu(self.branch3_bn2(self.branch3_conv2(x3)))
        x3 = self.branch3_pool2(x3)

        # 拼接三个分支的输出
        x_concat = torch.cat((x1, x2, x3), dim=2)
        
        # 展平层
        x = x.view(x_concat.size(0), -1)  
        x = F.relu(self.fc1(x))
        
        # 最终分类        
        output = self.fc2(x)
        return output

class MSCNN_2layers_adaptive_maxpool_3(nn.Module):
    """
    三分支网络，每个分支结构与原网络类似，最终输出拼接后分类。
    """
    def __init__(self, channels=3, num_classes=3):
        super(MSCNN_2layers_adaptive_maxpool_3, self).__init__()

        # 分支1
        self.branch1_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.branch1_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.AdaptiveMaxPool2d((1, 1))

        # 分支2
        self.branch2_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.branch2_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.AdaptiveMaxPool2d((1, 1))

        # 分支3
        self.branch3_conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.branch3_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=1)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.AdaptiveMaxPool2d((1, 1))

        # 全连接层，用于整合三个分支的输出
        self.fc1 = nn.Linear(in_features=64 * 3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # 分支1
        x1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        x1 = self.branch1_pool1(x1)
        x1 = F.relu(self.branch1_bn2(self.branch1_conv2(x1)))
        x1 = self.branch1_pool2(x1)  # 输出 (batch_size, 64, 1, 1)

        # 分支2
        x2 = F.relu(self.branch2_bn1(self.branch2_conv1(x)))
        x2 = self.branch2_pool1(x2)
        x2 = F.relu(self.branch2_bn2(self.branch2_conv2(x2)))
        x2 = self.branch2_pool2(x2)  # 输出 (batch_size, 64, 1, 1)

        # 分支3
        x3 = F.relu(self.branch3_bn1(self.branch3_conv1(x)))
        x3 = self.branch3_pool1(x3)
        x3 = F.relu(self.branch3_bn2(self.branch3_conv2(x3)))
        x3 = self.branch3_pool2(x3)  # 输出 (batch_size, 64, 1, 1)

        # 拼接三个分支的输出
        x_concat = torch.cat((x1, x2, x3), dim=1)  # 输出 (batch_size, 192, 1, 1)

        # 展平层
        x = x_concat.view(x_concat.size(0), -1)  # 展平为 (batch_size, 192)
        x = F.relu(self.fc1(x))  # 输入到 fc1，确保 fc1 的 in_features 为 192
        
        # 最终分类        
        output = self.fc2(x)  # 输入到 fc2
        return output

# %% adaptive
# %% effective
class CNN_2layers_adaptive_avgpool_2(nn.Module):
    """
    this model is identified as:
    2 convolution layers: c1 = c2 = 3, 1
    1 avgpool layers: p1 = 2, 2
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_2layers_adaptive_avgpool_2, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # 展平层
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        
        # 最终分类
        x = self.fc2(x)
        return x

class CNN_2layers_adaptive_maxpool_2(nn.Module):
    """
    this model is identified as:
    2 convolution layers: c1 = c2 = 3, 1
    1 maxpool layers: p1 = 2, 2
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_2layers_adaptive_maxpool_2, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_2layers_adaptive_avgpool_3(nn.Module):
    """
    this model is identified as:
    2 convolution layers: c1 = c2 = 3, 1
    1 avgpool layers: p1 = 3, 3
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_2layers_adaptive_avgpool_3, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# employed **********
class CNN_2layers_adaptive_maxpool_3(nn.Module):
    """
    this model is identified as:
    2 convolution layers: c1 = c2 = 3, 1
    1 maxpool layers: p1 = 3, 3
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_2layers_adaptive_maxpool_3, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# %%  not effective
class CNN_3layers_adaptive_avgpool_3(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    2 avgpool layers: p1 = p2 = 3, 3
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_adaptive_avgpool_3, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class CNN_3layers_adaptive_maxpool_3(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    2 maxpool layers: p1 = p2 = 3, 3
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_adaptive_maxpool_3, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class CNN_3layers_adaptive_avgpool_2(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    2 avgpool layers: p1 = p2 = 2, 2
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_adaptive_avgpool_2, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class CNN_3layers_adaptive_maxpool_2(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    2 maxpool layers: p1 = p2 = 2, 2
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_adaptive_maxpool_2, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# %% fixed
class CNN_3layers_avgpool(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    3 avgpool layers: p1 = p2 = p3 = 3, 3
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_avgpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 全连接层
        self.fc1 = nn.Linear(in_features=3*3*128, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class CNN_3layers_maxpool(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    3 maxpool layers: p1 = p2 = p3 = 3, 3
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_maxpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 全连接层
        self.fc1 = nn.Linear(in_features=3*3*128, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# %% 4 layers
class CNN_4layers_avgpool(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super(CNN_4layers_avgpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第四层卷积 + BatchNorm + 池化
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 全连接层
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
class CNN_4layers_maxpool(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super(CNN_4layers_maxpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第四层卷积 + BatchNorm + 池化
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 全连接层
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
