import torch
import numpy as np

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)

# a = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]
#
# tensor = torch.tensor(a)
# print(tensor)
# print(type(tensor))
# print(tensor.shape)
# print(tensor.reshape(1, 9))

# t = torch.Tensor()
# print(t.device)
# device = torch.device('cuda:0')
# print(device)

# t1 = torch.tensor([1, 2, 3])
# t2 = t1.cuda()
# print(t1.device)
# print(t2.device)

# print(torch.eye(2))
# print(torch.zeros(2))
# print(torch.ones(2))
# print(torch.rand(2))

t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]
])
# print(torch.tensor(t.shape).prod())
# print(t.numel())
# print(torch.tensor(t.shape).numel())
# print(t.reshape(1, 12))
# print(t.reshape(1, 12).squeeze())
# print(t.reshape(1, 12).unsqueeze(dim=0))

# t1 = torch.tensor([
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1]
# ])
# t2 = torch.tensor([
#     [2, 2, 2, 2],
#     [2, 2, 2, 2],
#     [2, 2, 2, 2],
#     [2, 2, 2, 2]
# ])
# t3 = torch.tensor([
#     [3, 3, 3, 3],
#     [3, 3, 3, 3],
#     [3, 3, 3, 3],
#     [3, 3, 3, 3]
# ])
# t = torch.stack([t1, t2, t3])
# print(t)
# print(t.reshape(1, -1))
# print(t.reshape(-1))
# print(t.flatten())
# # 对特定轴做降维打击
# print(t.flatten(start_dim=1))
# print(t.reshape(3, 16))
