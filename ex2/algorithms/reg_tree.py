#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:27:51 2021

@author: fabian
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from split_functions import RMS_residual

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
        self.depth = 0
        
        
class Const_regressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_attr_leaf=4, 
                 max_depth=20, 
                 smoothing=False, 
                 k=15.0, 
                 split_function="RMS",
                 prune_set = [],pruning=False):
        self.n_attr_leaf = n_attr_leaf
        self.max_depth = max_depth
        self.smoothing = smoothing
        self.k = k
        self.split_function = split_function
        self.prune_set = prune_set
        self.pruning=pruning
        
    def fit(self, X, y):
        data = np.hstack((X, y))
        self.root = self.create_regressor(data)
        
        if self.pruning:
            if self.prune_set != []:
                self.prune(self.prune_set[0], self.prune_set[1])
            else:
                self.prune(X, y)
        
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
            #print("calc split of {} points".format(data.shape[0]))
            if self.split_function=="RMS":
                node.error, node.dim_split, node.val = RMS_residual(data, node)
            elif self.split_function=="SDR":
                node.error, node.dim_split, node.val = SDR(data, node)
            if node.dim_split==-1 or node.error ==0: # no split found, or residual already 0 , make leaf
                node.type = 1 # leaf
                #print("Make it a leaf backup")
            else:
                mask_left  = data[:,node.dim_split]<node.val
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

                    
    def predict_vec(self, node, x):
        y = np.zeros((x.shape[0]))
        mask_left = x[:,node.dim_split] < node.val
        mask_right = np.logical_not(mask_left)
        
        if (node.left != None) and np.any(mask_left):
            if self.smoothing:
                y[mask_left] = (node.left.nval*self.predict_vec(node.left, x[mask_left,:])+\
                                self.k*node.coeff)/(node.left.nval+self.k)
            else:
                y[mask_left] = self.predict_vec(node.left, x[mask_left,:])
        elif np.any(mask_left):
            y[mask_left] = node.coeff
            
        if (node.right != None) and np.any(mask_right):
            if self.smoothing:
                y[mask_right] = (node.right.nval*self.predict_vec(node.right, x[mask_right,:])+\
                                self.k*node.coeff)/(node.right.nval+self.k)
            else:
                y[mask_right] = self.predict_vec(node.right, x[mask_right,:])
        elif np.any(mask_right):
            y[mask_right] = node.coeff
        return y
    
    def predict(self, X):
        return self.predict_vec(self.root, X)
    
    
    def prune(self, X, y):
        data = np.hstack((X, y))
        self.pruneby_abserror(self.root, data)
        
    def pruneby_abserror(self, node, data):
        mask_left = data[:,node.dim_split] < node.val
        mask_right = np.logical_not(mask_left)
        error_left = 0.0
        error_right = 0.0
        
        if (node.left != None) and np.any(mask_left):
            error_left  = ((data[mask_left,-1]-node.coeff)**2).sum()
            next_error_left = self.pruneby_abserror(node.left, data[mask_left,:])
            if next_error_left > error_left:
                node.left = None
                #print("prune left node at level",node.depth)
            else:
                error_left = next_error_left
        elif np.any(mask_left):
            error_left  = ((data[mask_left,-1]-node.coeff)**2).sum()
        if (node.right != None) and np.any(mask_right):
            error_right  = ((data[mask_right,-1]-node.coeff)**2).sum()
            next_error_right = self.pruneby_abserror(node.right, data[mask_right,:])
            if next_error_right > error_right:
                node.right = None
                #print("prune right node at level",node.depth)
            else:
                error_right = next_error_right
        elif np.any(mask_right):
           
            error_right  = ((data[mask_right,-1]-node.coeff)**2).sum()
        if (node.left == None) and (node.right == None):
            node.type = 1
        error_total = error_right+error_left
        if np.any(mask_left) and np.any(mask_right):
            error_total /= 2 # mean of left and right if both contribute
        return error_total
    