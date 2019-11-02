# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:39:13 2018

@author: Administrator
"""

from visdom import Visdom
import numpy as np

vis = Visdom()

# line plots
Y = np.linspace(-5, 5, 100)
vis.line(
    Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
    X=np.column_stack((Y, Y)),
    opts=dict(markers=False),
)
