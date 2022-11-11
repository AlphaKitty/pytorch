import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

print(len(train_set))
print(train_set.targets)
print(train_set.targets.bincount())

sample = next(iter(train_set))
print(len(sample))
print(type(sample))
image, label = sample
print(image.shape)
print(label)

print(image)
print(image.squeeze())
plt.imshow(image.squeeze(), cmap='gray')
# plt.show()
print('label:', label)

batch = next(iter(train_loader))
print(len(batch))
print(type(batch))
images, labels = batch
print(images.shape)
print(labels)

grid = torchvision.utils.make_grid(images, nrow=5)
plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()
print('labels:', labels)

print(grid[0].shape)
print(grid[1].shape)
print(grid[2].shape)
