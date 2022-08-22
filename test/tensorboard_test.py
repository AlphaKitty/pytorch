# 用tensorboard可视化网络的训练过程
# tensorboard --logdir "C:\Users\Administrator.DESKTOP-0HNMVEE\PycharmProjects\PyTorch01\test\runs"
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# tensorboard关键导入
from torch.utils.tensorboard import SummaryWriter

# 反向传播
import torch.optim as optim

from itertools import product
from run_builder import RunBuilder


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
train_set = torchvision.datasets.FashionMNIST(root='../data/FashionMNIST', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))

# batch_size = 100
# lr = 0.001
# batch_size_list = [100, 1000, 10000]
# lr_list = [.01, .001, .0001, .00001]
# 可调整的超参都放在这里然后接受遍历
parameters = dict(
    # 步长
    lr=[.01, .001],
    # 每批大小
    batch_size=[1000, 2000, 3000],
    # 是否打乱图片顺序
    shuffle=[True, False]
)
param_values = [v for v in parameters.values()]

# for batch_size in batch_size_list:
#     for lr in lr_list:
# *指示传入product的参数是字典里面的内容而不是字典本身 product返回的是字典的内部组合 一共2×2×2=8种组合
# for lr, batch_size, shuffle in product(*param_values):
# 经过前面两种循环的升级后最终用下面这种方式 输入是列表字典集合 输出是参数笛卡尔积集合
for run in RunBuilder.get_runs(parameters):
    print(str(run.lr) + " " + str(run.batch_size) + " " + str(run.shuffle))
    # comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'
    comment = f' -{run}'

    network = Network()
    # 把数据集以每批batch_size大小进行分割
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    # iter是返回list/tuple等可迭代对象的迭代器 next是返回迭代器的下一个元素 实际上就是遍历个过程
    # 这里实际上是只对每个超参组合的前batch_size张图片做了统计
    images, labels = next(iter(train_loader))
    # 组合图片成网格
    grid = torchvision.utils.make_grid(images)

    # 每次执行都会生成runs的一个命名目录
    tb = SummaryWriter(comment=comment)
    # 制作网格图像
    tb.add_image('images', grid)
    # 制作线形图
    tb.add_graph(network, images)

    # 重复
    for epoch in range(20):
        total_loss = 0
        total_correct = 0
        # 循环的是每批图片
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
            total_loss += loss.item() * run.batch_size
            # print(loss.item())
            total_correct += get_num_correct(preds, labels)

        # 损失
        tb.add_scalar('Loss', total_loss, epoch)
        # 正确识别数
        tb.add_scalar('Number Correct', total_correct, epoch)
        # 准确率
        tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

        # tb.add_histogram('conv1.bias', network.conv1.bias, loop)
        # tb.add_histogram('conv1.weight', network.conv1.weight, loop)
        # tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, loop)
        # 统计权重制作图表 取消硬编码
        for name, weight in network.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        print("epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)
    tb.close()
