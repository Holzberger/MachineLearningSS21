#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:58:14 2021

@author: fabian
"""
import numpy as np
import matplotlib.pyplot as plt

from treend import *
from reg_tree import *

def f1(x):
    return x**3 - 4*x**2 + x +1+ np.sin(10*x)

def f1_rand(n_samples,no_noise=False,seed=42):
    np.random.seed(seed)
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


def f2_rand(n_samples,no_noise=False,seed=42):
    xymin=-5
    xymax=5
    np.random.seed(seed)
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
    x,y = f1_rand(n_samples,no_noise=True)
    reg = M5regressor(smoothing=False, n_attr_leaf=8, max_depth=5, k=15.0,pruning=False,optimize_models=False,incremental_fit=False)
    reg.fit(x, y)
    X = np.linspace(-1,1,300)[:,None]
    if draw:
        Y = reg.predict(X)
        plt.plot(x,y,"o",alpha=0.5)
        plt.plot(X,Y,linewidth=2,color="red")
        
        # reg.smoothing=False
        # Y = reg.predict(X)
        # plt.plot(X,Y,linewidth=2,color="orange",linestyle=":")
        
        # x_test,y_test = f1_rand(100,no_noise=True)   
        # reg.prune(x_test, y_test)
        # Y = reg.predict(X)
        # plt.plot(X,Y,linewidth=2,color="green",linestyle="--")
        # plt.plot(X,f1(X),linewidth=2,color="black",alpha=0.5)
        
        x_test, y_test = f1_rand(n_samples,no_noise=False,seed=314)
        print(reg.score(x_test, y_test))
       
        
       
    
def test_2d(n_samples, draw=True):
    xymin=-5
    xymax=5
    
    x,y = f2_rand(n_samples,no_noise=True)
    
    #reg = M5regressor(smoothing=True, n_attr_leaf=15, max_depth=7, k=100.0)
    reg = M5regressor(smoothing=False, n_attr_leaf=4, max_depth=15, 
                  k=10.0,pruning=False,optimize_models=False,incremental_fit=False)
    reg.fit(x, y)
    
    x_test, y_test = f2_rand(400,no_noise=True)
    #reg.prune(x_test, y_test,optimize_models=True)
    
    #reg.prune(x, y, optimize_models=False)
    
    m=30
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
    x_test, y_test = f2_rand(100,no_noise=True,seed=314)
    print(reg.score(x_test, y_test))
    
def test_2d1(n_samples, draw=True):
    xymin=-5
    xymax=5
    
    x,y = f2_rand(n_samples)
    
    #reg = M5regressor(smoothing=True, n_attr_leaf=15, max_depth=7, k=100.0)
    reg = Const_regressor(n_attr_leaf=4, max_depth=30,smoothing=True,k=15)
    reg.fit(x, y)
    
    x_test, y_test = f2_rand(400,no_noise=True)
    #reg.prune(x_test, y_test,optimize_models=True)
    
    #reg.prune(x, y, optimize_models=False)
    
    m=30
    x_pos = np.linspace(xymin, xymax, m)
    y_pos = np.linspace(xymin, xymax, m)
    x_pos2d, y_pos2d = np.meshgrid(x_pos, y_pos)
    X = np.hstack((x_pos2d.reshape(m*m,1), y_pos2d.reshape(m*m,1)))
    Z = reg.predict(X).reshape(m,m)
    
    if draw:
        from matplotlib import cm
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x_pos2d, y_pos2d, Z, cmap=cm.coolwarm, linewidth=0,alpha=0.8)
        #surf = ax.scatter(x[:,0], x[:,1], y, cmap=cm.coolwarm,marker=".",alpha=0.3)
    # check score on some random testcases
    x_test, y_test = f2_rand(100,no_noise=True,seed=314)
    print(reg.score(x_test, y_test))
    
def test_1d1(n_samples,draw=True):
    x,y = f1_rand(n_samples,no_noise=False)
    reg = Const_regressor(n_attr_leaf=20, max_depth=15, smoothing=True,k=4)
    reg.fit(x, y)
    X = np.linspace(-1,1,300)[:,None]
    if draw:
        Y = reg.predict(X)
        plt.plot(x,y,".",alpha=0.5)
        plt.plot(X,Y,linewidth=2,color="red")
        
        plt.plot(X,f1(X),linewidth=2,color="black")
        
        x_test, y_test = f1_rand(n_samples,no_noise=False,seed=314)
        print(reg.score(x_test, y_test))
    
#test_1d(500)
#test_1d1(500)
test_2d(5000,draw=True)
#test_2d1(1000,draw=True)
