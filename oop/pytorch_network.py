import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 两层卷积层 out_channels是单图像深度
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5))

        # 三层线性层
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # 第一个卷积层
        t = self.conv1(t)
        # 0-1之间的取当前值 小于0的取0
        t = F.relu(t)
        # 对卷积层输出做最大池化 即在2*2的范围内取最大值作为该位置输出 目的是剔除冗余强化边界减少计算量增强边界识别
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 第二个卷积层
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 第一个线性层
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # 第二个线性层
        t = self.fc2(t)
        t = F.relu(t)

        # 输出层
        t = self.out(t)
        t = F.softmax(t, dim=1)
        return t
