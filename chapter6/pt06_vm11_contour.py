# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:43:26 2018

@author: Administrator
"""

from visdom import Visdom
import numpy as np

vis = Visdom()

x = np.tile(np.arange(1, 101), (100, 1))
y = x.transpose()
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))

vis.contour(X=X, opts=dict(colormap='Viridis'))
