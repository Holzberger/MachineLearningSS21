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
    y = noise + x**3 - 4*x**2 + x +1+ np.sin(10*x)
    return x[:,None],y[:,None]

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
    return x,y[:,None]

def test_1d(n_samples):
    x,y = f1(n_samples)
    data = np.hstack((x, y))
    root = create_M5(data)
    X = np.linspace(-1,1,100)[:,None]
    Y=np.array([predict(root, X[i,:][:,None],smoothing=True) for i in range(100)])
    plt.plot(x,y,".")
    plt.plot(X,Y,linewidth=2,color="red")
    Y=np.array([predict(root, X[i,:][:,None],smoothing=False) for i in range(100)])
    plt.plot(X,Y,linewidth=2,color="orange")
    
def test_2d(n_samples):
    x,y = f2_rand(n_samples)
    data = np.hstack((x, y))
    root = create_M5(data)
    m=25
    X = np.linspace(-1, 1, m)
    Y = np.linspace(-1, 1, m)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    for i in range(m):
        for j in range(m):
            x1 = np.array([X[i,j],Y[i,j]])[:,None].T
            Z[i,j] = predict(root, x1,smoothing=True)
    from matplotlib import cm
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha=0.8)
    surf = ax.scatter(x[:,0], x[:,1], y, cmap=cm.coolwarm,marker=".")

#test_1d(500)
test_2d(500)

# # # draw splits
# splits = print_split(root)
# for split in splits:
#     plt.axvline(x=split,color="orange",linestyle="--",linewidth=1)
