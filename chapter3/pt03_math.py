#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 21:29:42 2018

@author: pc
"""
import torch
c=torch.abs(torch.FloatTensor([-1, -2, 3]))
a = torch.randn(4)
print(a)
b=torch.add(a, 20)
print(b)