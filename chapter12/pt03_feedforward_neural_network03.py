#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:20:54 2017

@author: pc
"""
# -*- coding: utf-8 -*-

import torch
import torch
import pandas as pd
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
df=pd.read_excel(r"/home/pc/上证指数数据.xlsx")

df1=df.iloc[:100,3:6].values
xtrain_features=torch.FloatTensor(df1)
df2=df["涨跌"].astype(float)
xtrain_labels=torch.FloatTensor(df2[:100])
xtrain=torch.unsqueeze(xtrain_features,dim=1)
ytrain=torch.unsqueeze(xtrain_labels,dim=1)
x, y = torch.autograd.Variable(xtrain), Variable(ytrain)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
net = Net(input_size=4, hidden_size=100, num_classes=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
for epoch in range(100000):
    inputs =x
    target =y
    out =net(inputs)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print('Epoch[{}], loss: {:.6f}'.format(epoch+1,loss.data[0]))

