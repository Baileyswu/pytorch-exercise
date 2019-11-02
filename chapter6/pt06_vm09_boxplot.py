# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:42:41 2018

@author: Administrator
"""

from visdom import Visdom
import numpy as np

vis = Visdom()

# boxplot
X = np.random.rand(100, 2)
X[:, 1] += 2

vis.boxplot(
    X=X,
    opts=dict(legend=['Men', 'Women'])
)
