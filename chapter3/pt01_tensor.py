#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 21:04:47 2018

@author: pc
"""

import torch
z = torch.Tensor(4, 5)
print(z)
y = torch.rand(4, 5)#产生一个4行5列的矩阵
print(z + y)
print(torch.add(z, y))
b = z.numpy()
print(b)