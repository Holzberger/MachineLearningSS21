#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:29:52 2021

@author: fabian
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
class Node():
    root_std = 0
    def __init__(self):
        self.left  = None
        self.right = None
        self.dim_split = 0
        self.nval  = 0 
        self.val   = 0
        self.node_std = 0
        self.type  = 0 # 0 interior, 1 leaf
        self.coeffs = [0]
        self.error= -1
        self.depth =0
    def predictby_nodemodel(self, x):
        return x.dot(self.coeffs[:-1]) + self.coeffs[-1]
        

class M5regressor(BaseEstimator, RegressorMixin):
    def __init__(self, smoothing=True, n_attr_leaf=4, max_depth=20, k=15.0):
        self.smoothing = smoothing
        self.n_attr_leaf = n_attr_leaf
        self.max_depth = max_depth
        self.k = k
        
    
    def fit(self, X, y):
        data = np.hstack((X, y))
        self.root = self.create_M5(data)
        return self
    
    def predict(self, X):
        return self.predict_vec(self.root, X, smoothing=self.smoothing)
        
    def create_M5(self, X):
        x = np.copy(X)
        root = Node()
        root.root_std = np.std(x[:, -1])
        self.split(root, x)   
        return root
    
    def SDR(self, data, node):
        n_T, n_attr = data.shape
        n_attr -= 1 
        T   = data[:,-1]
        sdT = node.node_std
        attr = data[:, :-1]
        attr_means = np.mean(attr, axis=0) 
        mask_T1 = attr <= attr_means
        SDR_max = -1.0
        for n in range(n_attr):
            T1 = T[mask_T1[:,n]]
            T2 = T[np.logical_not(mask_T1[:,n])]
            SDR_attr = np.abs(sdT - (T1.shape[0]*np.std(T1) + 
                                     T2.shape[0]*np.std(T2))/n_T)
            if SDR_attr > SDR_max:
                SDR_max = SDR_attr
                n_SDR_max = n
        return data[mask_T1[:,n_SDR_max],:], \
               data[np.logical_not(mask_T1[:,n_SDR_max]),:], \
               attr_means[n_SDR_max], \
               n_SDR_max
    
    def split(self, node, data):
        # setup linear model in all nodes
        x_coeffs = np.hstack([data[:,:-1], np.ones((data.shape[0],1))])
        node.coeffs = np.linalg.lstsq(x_coeffs, data[:,-1], rcond=None)[0]
        node.nval = data.shape[0]
        node.node_std = np.std(data[:,-1])
        if (node.nval < self.n_attr_leaf) or (node.node_std<node.root_std*0.05) or node.depth>self.max_depth:
            node.type = 1 # leaf
        else:
            data_left, data_right, mean, dim_split = self.SDR(data, node)
            node.type = 0 # interiour
            node.left  = Node()
            node.right = Node()
            node.left.depth = node.depth+1
            node.right.depth = node.depth+1
            node.val  = mean
            node.dim_split = dim_split
            self.split(node.left, data_left)
            self.split(node.right, data_right)
            
    def predict_vec(self, node, x, smoothing=False):
        y = np.zeros((x.shape[0]))
        mask_left = x[:,node.dim_split] <= node.val
        mask_right = np.logical_not(mask_left)
        
        if (node.left != None) and np.any(mask_left):
            if smoothing:
                y[mask_left] = (node.left.nval*self.predict_vec(node.left, x[mask_left,:])+\
                                self.k*node.predictby_nodemodel(x[mask_left,:]))/(node.left.nval+self.k)
            else:
                y[mask_left] = self.predict_vec(node.left, x[mask_left,:])
        elif np.any(mask_left):
            y[mask_left] = node.predictby_nodemodel(x[mask_left,:])
            
        if (node.right != None) and np.any(mask_right):
            if smoothing:
                y[mask_right] = (node.right.nval*self.predict_vec(node.right, x[mask_right,:])+\
                                self.k*node.predictby_nodemodel(x[mask_right,:]))/(node.right.nval+self.k)
            else:
                y[mask_right] = self.predict_vec(node.right, x[mask_right,:])
        elif np.any(mask_right):
            y[mask_right] = node.predictby_nodemodel(x[mask_right,:])
        return y
        
    def prune(self, X, y):
        data = np.hstack((X, y))
        self.pruneby_abserror(self.root, data)
        
    def pruneby_abserror(self, node, data):
        mask_left = data[:,node.dim_split]<=node.val
        mask_right = np.logical_not(mask_left)
        error_left = 0.0
        error_right = 0.0
        if (node.left != None) and np.any(mask_left):
            error_left  = np.mean(np.abs(node.predictby_nodemodel(data[mask_left,:-1])-data[mask_left,-1]))
            next_error_left = self.pruneby_abserror(node.left, data[mask_left,:])
            if next_error_left > error_left:
                node.left = None
            else:
                error_left = next_error_left
        elif np.any(mask_left):
            error_left  = np.mean(np.abs(node.predictby_nodemodel(data[mask_left,:-1])-data[mask_left,-1]))
        
        if (node.right != None) and np.any(mask_right):
            error_right = np.mean(np.abs(node.predictby_nodemodel(data[mask_right,:-1])-data[mask_right,-1]))
            next_error_right = self.pruneby_abserror(node.right, data[mask_right,:])
            if next_error_right > error_right:
                node.right = None
            else:
                error_right = next_error_right
        elif np.any(mask_right):
            error_right = np.mean(np.abs(node.predictby_nodemodel(data[mask_right,:-1])-data[mask_right,-1]))
        
        if (node.left == None) and (node.right == None):
            node.type = 1
        error_total = error_right+error_left
        if np.any(mask_left) and np.any(mask_right):
            error_total /= 2 # mean of left and right if both contribute
        return error_total
        
        
    