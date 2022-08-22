from pytorch_network import Network
from pytorch_run_manager import RunManager
from pytorch_run_builder import RunBuilder

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# 反向传播
import torch.optim as optim

# 训练集
train_set = torchvision.datasets.FashionMNIST(root='../data/FashionMNIST', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
params = OrderedDict(
    lr=[.01, .001],
    batch_size=[1000, 2000],
    shuffle=[True, False]
)

m = RunManager()
for run in RunBuilder.get_runs(params):
    network = Network()
    loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, loader)
    for epoch in range(5):
        m.begin_epoch()
        for batch in loader:
            images = batch[0]
            labels = batch[1]
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m.track_loss(loss)
            m.track_num_correct(preds, labels)
        m.end_epoch()
    m.end_run()
m.save('results')
