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
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n_samples)
    noise = np.random.normal(0, 0.5, n_samples)
    if no_noise:
        noise_coeff = 0.0
    else:
        noise_coeff = 1.0
    y = noise*noise_coeff + f1(x)
    return x[:,None],y[:,None]


def f2(x):
    return np.sin(x[:,0])*np.cos(x[:,1])#x[:,0]**2+x[:,1]**2+10


def f2_rand(n_samples,no_noise=False):
    xymin=-5
    xymax=5
    np.random.seed(42)
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
    reg = M5regressor(smoothing=True, n_attr_leaf=4, max_depth=3, k=20.0)
    reg.fit(x, y)
    X = np.linspace(-1,1,300)[:,None]
    if draw:
        reg.smoothing=True
        Y = reg.predict(X)
        plt.plot(x,y,".",alpha=0.5)
        plt.plot(X,Y,linewidth=2,color="red")
        
        reg.smoothing=True
        Y = reg.predict(X)
        plt.plot(X,Y,linewidth=2,color="orange")
        
        x_test,y_test = f1_rand(100,no_noise=True)   
        reg.prune(x_test, y_test)
        Y = reg.predict(X)
        plt.plot(X,Y,linewidth=2,color="green")
        plt.plot(X,f1(X),linewidth=2,color="black")
        
        reg.prune(x_test, y_test)
        print(reg.score(x_test, y_test))
        
    
def test_2d(n_samples, draw=True):
    xymin=-5
    xymax=5
    
    x,y = f2_rand(n_samples)
    
    reg = M5regressor(smoothing=True, n_attr_leaf=10, max_depth=9, k=20.0)
    reg.fit(x, y)
    
    x_test, y_test = f2_rand(100,no_noise=True)
    reg.prune(x_test, y_test)
    
    m=25
    x_pos = np.linspace(xymin, xymax, m)
    y_pos = np.linspace(xymin, xymax, m)
    x_pos2d, y_pos2d = np.meshgrid(x_pos, y_pos)
    X = np.hstack((x_pos2d.reshape(m*m,1), y_pos2d.reshape(m*m,1)))
    Z = reg.predict(X).reshape(m,m)
    
    if draw:
        from matplotlib import cm
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x_pos2d, y_pos2d, Z, cmap=cm.coolwarm, linewidth=0,alpha=0.8)
        surf = ax.scatter(x[:,0], x[:,1], y, cmap=cm.coolwarm,marker=".",alpha=0.3)
    # check score on some random testcases
    x_test, y_test = f2_rand(100,no_noise=True)
    print(reg.score(x_test, y_test))
    
    
    
    
    
    
test_1d(500)
#test_2d(1200,draw=True)

