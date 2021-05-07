#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:58:14 2021

@author: fabian
"""
import numpy as np
import matplotlib.pyplot as plt

from tree1d import *

def f1(n_samples):
    x = np.random.uniform(-1, 1, n_samples)
    noise = np.random.normal(0, 0.5, n_samples)
    y = noise+x**3 - 4*x**2 + x +1+ np.sin(10*x)
    return x,y

x,y = f1(100)

plt.plot(x,y,".")

data = np.stack((x, y)).T

root = create_M5(data)

# draw model
X = np.linspace(-1,1,100)
Y=np.array([predict(root, elem) for elem in X])
plt.plot(X,Y,linewidth=2,color="red")

# draw splits
splits = print_split(root)
for split in splits:
    plt.axvline(x=split,color="orange",linestyle="--",linewidth=1)
