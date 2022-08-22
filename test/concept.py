# 区分concat和stack的区别
import torch

t1 = torch.tensor([1, 1, 1])
t2 = torch.tensor([2, 2, 2])
t3 = torch.tensor([3, 3, 3])

# cat是在已有维度上对张量叠加
print(torch.cat((t1, t2, t3), dim=0))
# stack是在新的维度上对张量堆叠
print(torch.stack((t1, t2, t3), dim=1))

# 在tensorflow和numpy中分别也有对应的方法 stack都叫stack 而cat分别叫concat和concat????
