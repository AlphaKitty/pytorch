# 增量训练
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# 反向传播
import torch.optim as optim


# 定义两个卷积层 三个全连接层的网络
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


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# 训练集
train_set = torchvision.datasets.FashionMNIST(
    root='../data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
# 实例化网络
# network = Network()
network = torch.load('C:\\Users\\Administrator.DESKTOP-0HNMVEE\\Desktop\\bao1\\network.pth')
# 步长为0.01搭配网络权重的优化器
optimizer = optim.Adam(network.parameters(), lr=0.00005)
# 把数据集以每批100进行分割
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=100
)
# 如果设置False则不计算梯度
torch.set_grad_enabled(True)

for loop in range(1):
    total_loss = 0
    total_correct = 0
    # 每个batch就是100张带标签的图片
    # 可通过以下watch调试
    # network.conv1.weight.grad.max().item()
    # network.conv1.weight.max().item()
    # images.shape
    # labels.shape
    # preds.shape
    # loss.item()
    # total_loss
    # total_correct
    for batch in train_loader:
        # 图片和标签分开
        images, labels = batch

        # 图片输入到网络 经过网络加工后获取预测
        preds = network(images)
        # 用交叉熵(差平方和)处理预测和标签获取损失
        # 熵是最优策略下实现某概率事件的期望 熵越大系统不确定性越大
        # 交叉熵是给定策略下实现某概率事件的期望 交叉熵越接近熵说明给定策略越接近最优
        # 交叉熵总是大于等于熵 优化就是让交叉熵更接近熵
        # https://baijiahao.baidu.com/s?id=1618702220267847958&wfr=spider&for=pc
        loss = F.cross_entropy(preds, labels)

        # 清空旧梯度
        optimizer.zero_grad()
        # 计算偏导 更新的是network.conv.weight.grad
        loss.backward()
        # 更新权重 更新的是network.conv.weight
        optimizer.step()

        total_loss += loss.item()
        # print(loss.item())
        total_correct += get_num_correct(preds, labels)

    print("epoch:", loop, "total_correct:", total_correct, "loss:", total_loss)
    print(total_correct / len(train_set))

torch.save(network, 'C:\\Users\\Administrator.DESKTOP-0HNMVEE\\Desktop\\bao1\\network.pth')
