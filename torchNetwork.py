import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # 输出通道(特征图)数=滤波器(卷积核)数
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5))

        # 线性层也就是全连接层 输入的是in_features个元素的in_features维特征张量 输出的是out_features个元素的out_features维特征张量
        # 两个参数传递后Linear会创建一个in_features行out_features列的权重矩阵
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        self.layer = None

    def forward(self, t):
        t = self.layer(t)
        return t


network = Network()
print(network)
print(network.conv1)
# print(network.conv1.weight)
# 第一个卷积层: 6个5×5的滤波器对1个输入通道做卷积 也可以看做每个滤波器深度是1
print(network.conv1.weight.shape)
# 第二个卷积层: 12个5×5的滤波器对6个输入通道做卷积 也可以看做每个滤波器深度是6
print(network.conv2.weight.shape)
print(network.fc1.weight.shape)
print(network.fc2.weight.shape)
print(network.out.weight.shape)

in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
weight_matrix = torch.tensor([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6]
], dtype=torch.float32)
print(weight_matrix.matmul(in_features))
fc = nn.Linear(in_features=4, out_features=3)
print(fc(in_features))
