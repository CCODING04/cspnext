import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                      MultiStepLR)

# class Model(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.m = nn.Linear(in_features, out_features)
    
#     def forward(self, x):
#         return self.m(x)
base_lr = 0.1
model = torch.nn.Linear(1, 1)
# model = Model(1, 1)
dataset = [torch.randn((1, 1, 1)) for _ in range(20)]
optimizer = SGD(model.parameters(), base_lr, momentum=0.9)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
# scheduler = MultiStepLR(optimizer, milestones=[5, 9], gamma=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=12*len(dataset), eta_min=0.05 * base_lr)

lr_lst = []
for epoch in range(12):
    for data in dataset:
        optimizer.zero_grad()
        output = model(data)
        loss = 1 - output
        loss.backward()
        optimizer.step()
        lr = scheduler.get_last_lr()
        lr_lst.append(lr[0])
        scheduler.step()

plt.plot(list(range(len(lr_lst))), lr_lst, "-r")
# for idx, s in enumerate(lr_lst):
#     plt.text(idx, lr_lst[idx], s, fontsize=10)
plt.savefig("out.png")
