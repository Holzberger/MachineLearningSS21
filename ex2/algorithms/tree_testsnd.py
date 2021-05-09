#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:58:14 2021

@author: fabian
"""
import numpy as np
import matplotlib.pyplot as plt

from treend import *

def f1(n_samples):
    x = np.random.uniform(-1, 1, n_samples)
    noise = np.random.normal(0, 1, n_samples)
    y = noise+x**3 - 4*x**2 + x +1+ np.sin(10*x)
    return x,y

def Fn(x):
    return x[:,0]**3 - 4*x[:,1]**2 + x[:,2] +1+ np.sin(10*x[:,3])

def fn(n_samples):
    x = np.zeros((n_samples, 4))
    for dim in range(4):
        x[:, dim]     = np.random.uniform(-1, 1, n_samples)
    noise = np.random.normal(0, 2, n_samples)
    y = noise + Fn(x)
    return x,y


def f2(x):
    return x[:,0]**2+x[:,1]**2+10

def f2_rand(n_samples):
    x = np.zeros((n_samples, 2))
    for dim in range(2):
        x[:, dim]     = np.random.uniform(-1, 1, n_samples)
    noise = np.random.normal(0, 0.1, n_samples)
    y = noise + f2(x)
    return x,y

x,y = f2_rand(1000)





#np.std(a, dtype=np.float64)

data = np.hstack((x, y[:,None]))

root = create_M5(data)



m=20
X = np.linspace(-1, 1, m)
Y = np.linspace(-1, 1, m)
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X)

for i in range(m):
    for j in range(m):
        x1 = np.array([X[i,j],Y[i,j]])[:,None].T
        Z[i,j] = predict(root, x1)
from matplotlib import cm
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha=0.5)
surf = ax.scatter(x[:,0], x[:,1], y, cmap=cm.coolwarm)

#splits = print_split(root)

# # draw model
# plt.plot(x,y,".")
# X = np.linspace(-1,1,100)[:,None]
# Y=np.array([predict(root, X[i,:][:,None]) for i in range(100)])
# plt.plot(X,Y,linewidth=2,color="red")

# # # draw splits
# splits = print_split(root)
# for split in splits:
#     plt.axvline(x=split,color="orange",linestyle="--",linewidth=1)
