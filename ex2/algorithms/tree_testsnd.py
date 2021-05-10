#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:58:14 2021

@author: fabian
"""
import numpy as np
import matplotlib.pyplot as plt

from treend import *

def f1(x):
    return x**3 - 4*x**2 + x +1+ np.sin(10*x)

def f1_rand(n_samples,no_noise=False):
    x = np.random.uniform(-1, 1, n_samples)
    noise = np.random.normal(0, 0.5, n_samples)
    if no_noise:
        noise_coeff = 0.0
    else:
        noise_coeff = 1.0
    y = noise*noise_coeff + f1(x)
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
    return np.sin(x[:,0])*np.cos(x[:,1])#x[:,0]**2+x[:,1]**2+10


def f2_rand(n_samples,no_noise=False):
    xymin=-5
    xymax=5
    x = np.zeros((n_samples, 2))
    for dim in range(2):
        x[:, dim]     = np.random.uniform(xymin, xymax, n_samples)
    if no_noise:
        noise_coeff = 0.0
    else:
        noise_coeff = 1.0
    noise = np.random.normal(0, 0.1, n_samples)*noise_coeff
    y = noise + f2(x)
    return x,y[:,None]

def test_1d(n_samples,draw=True):
    x,y = f1_rand(n_samples,no_noise=False)
    data = np.hstack((x, y))
    root = create_M5(data)
    X = np.linspace(-1,1,300)[:,None]
    if draw:
        Y=predict_vec(root,X,smoothing=True)
        plt.plot(x,y,".",alpha=0.5)
        plt.plot(X,Y,linewidth=2,color="red")
        Y=predict_vec(root,X,smoothing=False)
        plt.plot(X,Y,linewidth=2,color="orange")
        x_test,y_test = f1_rand(100,no_noise=True)   
        data = np.hstack((x_test, y_test))
        prune(root, data)
        Y=predict_vec(root,X,smoothing=True)
        plt.plot(X,Y,linewidth=2,color="green")
        plt.plot(X,f1(X),linewidth=2,color="black")
        
    
def test_2d(n_samples, draw=True, show_splits=False):
    xymin=-5
    xymax=5
    x,y = f2_rand(n_samples)
    data = np.hstack((x, y))
    root = create_M5(data)
    
    x_test, y_test = f2_rand(300,no_noise=True)
    data = np.hstack((x_test, y_test))
    prune(root, data)
    
    m=25
    XX = np.linspace(xymin, xymax, m)
    YY = np.linspace(xymin, xymax, m)
    X, Y = np.meshgrid(XX, YY)
    Z = np.zeros_like(X)
    data = np.hstack((X.reshape(m*m,1), Y.reshape(m*m,1)))
    Z = predict_vec(root,data,smoothing=False).reshape(m,m)
    if draw:
        from matplotlib import cm
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False,alpha=0.8)
        surf = ax.scatter(x[:,0], x[:,1], y, cmap=cm.coolwarm,marker=".")
    if show_splits:
        splits = print_split(root)
    

#test_1d(500)
test_2d(1200,draw=True)


