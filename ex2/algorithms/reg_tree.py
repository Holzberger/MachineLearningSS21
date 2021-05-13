#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:27:51 2021

@author: fabian
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
class Node():
    dims     = 0
    def __init__(self):
        self.left  = None
        self.right = None
        self.dim_split = 0
        self.nval  = 0 
        self.val   = 0
        self.type  = 0 # 0 interior, 1 leaf
        self.coeff = 0.0
        self.error= -1
        self.depth =0
        
        
class Const_regressor(BaseEstimator, RegressorMixin):
    def __init__(self,n_attr_leaf=4, max_depth=20):
        self.n_attr_leaf = n_attr_leaf
        self.max_depth = max_depth
        
    def fit(self, X, y):
        data = np.hstack((X, y))
        self.root = self.create_regressor(data)
        return self
    
    def create_regressor(self, X):
        x = np.copy(X)
        root = Node()
        self.split(root, x)   
        return root
    
    def split(self, node, data):
        node.coeff = np.mean(data[:,-1])
        node.nval = data.shape[0]
        if (node.nval < self.n_attr_leaf) or\
            (node.depth>self.max_depth) or\
            (np.max(data[:,:-1],axis=0)-np.min(data[:,:-1],axis=0)<1e-12).all():
            node.type = 1 # leaf
        else:
            #print(node.coeffs)
            print("calc RMS of {} points".format(data.shape[0]))
            node.error, node.dim_split, node.val = self.rRMS(data, node)
            mask_left  = data[:,node.dim_split]<=node.val
            mask_right = np.logical_not(mask_left)
            data_left = data[mask_left,:]
            data_right = data[mask_right,:]
            node.type = 0 # interiour
            node.left  = Node()
            node.right = Node()
            node.left.depth = node.depth+1
            node.right.depth = node.depth+1
            self.split(node.left, data_left)
            self.split(node.right, data_right)
           

        
    def rRMS(self, data, node):
        n_T, n_attr = data.shape
        n_attr -= 1 
        T   = data[:,-1]
        attr = data[:, :-1]
        res_min = 1e20
        n_res_min = -1
        split_val_min = -1
        for n in range(n_attr):
            unique_attrs = np.unique(data[:,n])
            for split_val in unique_attrs[1::10]:
                mask0 = data[:,n]<=split_val
                mask1 = np.logical_not(mask0)
                res = np.sum((np.mean(T[mask0])-T[mask0])**2) + np.sum((np.mean(T[mask1])-T[mask1])**2)
                if res<res_min:
                    res_min=res
                    n_res_min = n
                    split_val_min = split_val
        return res_min, n_res_min, split_val_min
                    
    def predict_vec(self, node, x):
        y = np.zeros((x.shape[0]))
        mask_left = x[:,node.dim_split] <= node.val
        mask_right = np.logical_not(mask_left)
        
        if (node.left != None) and np.any(mask_left):
            y[mask_left] = self.predict_vec(node.left, x[mask_left,:])
        elif np.any(mask_left):
            y[mask_left] = node.coeff
            
        if (node.right != None) and np.any(mask_right):
            y[mask_right] = self.predict_vec(node.right, x[mask_right,:])
        elif np.any(mask_right):
            y[mask_right] = node.coeff
        return y
    
    def predict(self, X):
        return self.predict_vec(self.root, X)
    