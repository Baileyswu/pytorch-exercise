#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 21:31:54 2018

@author: pc
"""
import torch
a = torch.randn(1, 3)
print(a)

b=torch.mean(a)
print(b)

c = torch.randn(4, 4)
print(c)



d= torch.mean(c, 1)

print(d)