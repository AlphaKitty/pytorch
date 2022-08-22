# C:\Users\Administrator.DESKTOP-0HNMVEE\PycharmProjects\PyTorch01\test\runs
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# 关键导入
from torch.utils.tensorboard import SummaryWriter

# 反向传播
import torch.optim as optim

from itertools import product

parameters = dict(
    lr=[.01, .001],
    batch_size=[10, 100, 1000],
    shuffle=[True, False]
)

# 这里其实是一个for循环的快速写法
param_values = [v for v in parameters.values()]
print(param_values)

# 这个*表示传值是字典的三个值做参数 而不是字典本身
for lr, batch_size, shuffle in product(*param_values):
    print(lr, batch_size, shuffle)

print(torch.__version__)
print(torchvision.__version__)


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
# 如果设置False则不计算梯度
torch.set_grad_enabled(True)

batch_size = 100
lr = 0.001
shuffle = True
batch_size_list = [100, 1000, 10000]
lr_list = [.01, .001, .0001, .00001]

for batch_size in batch_size_list:
    for lr in lr_list:
        network = Network()
        # 把数据集以每批100进行分割
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            # 打乱批次
            shuffle=shuffle
        )
        optimizer = optim.Adam(network.parameters(), lr=lr)

        images, labels = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images)

        # 批大小和步长参与名称
        # comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'
        # comment = ''
        # tb = ''
        # 错误写法 不是在这加
        for lr, batch_size, shuffle in product(*param_values):
            comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'
            tb = SummaryWriter(comment=comment)
        tb.add_image('images', grid)
        tb.add_graph(network, images)

for loop in range(3):
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
        loss = F.cross_entropy(preds, labels)

        # 清空旧梯度
        optimizer.zero_grad()
        # 计算偏导 更新的是network.conv.weight.grad
        loss.backward()
        # 更新权重 更新的是network.conv.weight
        optimizer.step()

        # 和之前不同的 乘batch_size是因为loss是平均损失 相乘之后更有可比性
        total_loss += loss.item() * batch_size
        # print(loss.item())
        total_correct += get_num_correct(preds, labels)

    tb.add_scalar('Loss', total_loss, loop)
    tb.add_scalar('Number Correct', total_correct, loop)
    tb.add_scalar('Accuracy', total_correct / len(train_set), loop)

    # tb.add_histogram('conv1.bias', network.conv1.bias, loop)
    # tb.add_histogram('conv1.weight', network.conv1.weight, loop)
    # tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, loop)
    # 取消硬编码
    for name, weight in network.named_parameters():
        tb.add_histogram(name, weight, loop)
        tb.add_histogram(f'{name}.grad', weight.grad, loop)

    print("epoch:", loop, "total_correct:", total_correct, "loss:", total_loss)
    print(total_correct / len(train_set))

tb.close()
