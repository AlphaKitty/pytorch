import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# 反向传播
import torch.optim as optim


# 卷积的本质实际上是输入的每个元素灰度值和权重的乘积(表示该元素在当前卷积层下有多被重视) 然后求和 构成卷集层的一个元素
# 学习的本质实际上是调整这些权重和转置
# 反向传播基于代价函数 损失函数就是每个样本所有预测值和标签值的差平方和 代价函数就是最后取所有样本的平均值 然后想办法让该值尽量小
# 函数的梯度指出了函数最陡的方向 沿梯度的反方向走是局部最小值的最短路径 所以1计算梯度 2沿梯度反方向走一小段 3重复
# 正向传播输入是图像像素值输出是分类 代价函数输入是权重输出是代价评分 梯度输入是权重输出是权重优化
# 卷积层输出图像宽度公式:M = (N+2*padding-kernel_size)/stride-1
# 其中N是输入宽高 padding是填充宽度(一般对称填充所以乘二) kernel_size是卷积核宽高 stride是步长
# 定义网络类 继承Module基类
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


print(torch.__version__)
print(torchvision.__version__)
# 设置输出行宽
torch.set_printoptions(linewidth=120)
# 关闭损失函数梯度计算 开始训练的时候再打开
# torch.set_grad_enabled(False)

# 加载样本数据集
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

network = Network()

# # 单个图像的测试
# sample = next(iter(train_set))
# image, label = sample
# # 把单个图像变为批次
# pred = network(image.unsqueeze(0))
# print(F.softmax(pred, dim=1).max())
# print(F.softmax(pred, dim=1).sum())

# 批处理图像
data_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=100
)
batch = next(iter(data_loader))
images, labels = batch
# (batch_size(多少个图像), input_channels(多少个色彩通道), height(高度), width(宽度))
# print(images.shape)
# 每个图像对应一个标签
# print(labels.shape)
# data_loader出来的默认就是批 所以不需要在unsqueeze

preds = network(images)
# # 每个图像的类别预测
# print(preds)
# # 找到每个图像预测的最大值的索引
# print(preds.argmax(dim=1))
# # 正确预测
# print(labels)

# 计算损失(预测结果和真实结果之间的差别)
loss = F.cross_entropy(preds, labels)
# print(loss.item())
# # 调用反向函数计算梯度
# print(network.conv1.weight.grad)
# 计算偏导
loss.backward()
# print(network.conv1.weight.grad.shape)


# Adam/SGD lr是学习速率(learn rate) 也就是沿梯度反方向每次走的步长
optimizer = optim.Adam(network.parameters(), lr=0.01)
print(loss.item())
print(get_num_correct(preds, labels))
# 更新权重
optimizer.step()
preds = network(images)
loss = F.cross_entropy(preds, labels)
print(loss.item())
print(get_num_correct(preds, labels))



