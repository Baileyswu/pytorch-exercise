#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 21:34:42 2018

@author: pc
"""
from torch import autograd 
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
m = nn.Conv1d(16, 33, 3, stride=2)
inputs = autograd.Variable(torch.randn(20, 16, 50))
output = m(inputs)
print(output)
m = nn.MaxPool1d(3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50))
output = m(input)
print(output)