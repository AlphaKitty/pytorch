# 构建混淆矩阵 可以看预测和实际之间的差别
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
# 自己定义的方法
from plotcm import plot_confusion_matrix


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


# 训练集
train_set = torchvision.datasets.FashionMNIST(
    root='../data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


# print(len(train_set))
# print(len(train_set.targets))


def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
    return all_preds


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


network = torch.load('C:\\Users\\Administrator.DESKTOP-0HNMVEE\\Desktop\\bao1\\network.pth')

# with后面跟的是临界变量(比如锁 资源等 相当于try/finally) 在这里表示with包裹的代码都不计算梯度 也可以在方法上用注解@torch.no_grad()
# 这里关闭梯度追踪是因为构建混淆矩阵只需要预测张量和标签张量 不需要训练过程
with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

# 是否需要为张量计算梯度 需要训练为True 否则False
print('train_preds: ' + str(train_preds.requires_grad))
# 梯度值
print('grad: ' + str(train_preds.grad))
# 梯度函数原函数 方便计算梯度
print('grad_fn: ' + str(train_preds.grad_fn))

preds_correct = get_num_correct(train_preds, train_set.targets)
print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(train_set))

# 标签张量
print("标签张量: " + str(train_set.targets))
# 预测张量
print("预测张量: " + str(train_preds.argmax(dim=1)))

# 标签张量和预测张量拼接
stacked = torch.stack(
    (
        train_set.targets,
        train_preds.argmax(dim=1)
    ),
    dim=1
)
print(stacked.shape)
print(stacked)
print(stacked[0].tolist())

cmt = torch.zeros(10, 10, dtype=torch.int32)

# 遍历一一映射好的预测/标签张量对 在10×10的二维坐标里对应位置+1 最终训练好的网络结构会呈现左上到右下的对角线
for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1
print(cmt)

# [5642,    4,   60,   63,   18,    0,  182,    0,   31,    0],
# [   2, 5938,    2,   44,    5,    0,    6,    0,    3,    0],
# [  55,    1, 5534,   41,  206,    1,  152,    0,   10,    0],
# [  49,   11,   22, 5769,   87,    1,   55,    0,    6,    0],
# [   6,    4,  170,   99, 5574,    0,  140,    0,    7,    0],
# [   1,    0,    0,    0,    0, 5915,    0,   61,    7,   16],
# [ 411,    7,  207,   82,  215,    1, 5054,    0,   23,    0],
# [   0,    0,    0,    0,    0,   27,    0, 5863,    5,  105],
# [   9,    2,   13,   12,   13,    6,   16,    1, 5925,    3],
# [   0,    0,    0,    0,    1,   12,    0,  100,    4, 5883]

# 这里cm和cmt做的是一样的工作
cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
print(type(cm))
print(cm)

names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot')
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, names)
plt.show()
