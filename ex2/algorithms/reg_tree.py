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
    def __init__(self,n_attr_leaf=4, max_depth=20, 
                 smoothing=False, k=15.0):
        self.n_attr_leaf = n_attr_leaf
        self.max_depth = max_depth
        self.smoothing = smoothing
        self.k = k
        
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
            print("rms, dim, split",node.error, node.dim_split, node.val)
            if node.dim_split==-1 or node.error ==0: # no split found, or residual already 0 , make leaf
                node.type = 1 # leaf
                print("Make it a leaf backup")
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
           

        
    def rRMS(self, data, node):
        n_T, n_attr = data.shape
        n_attr -= 1 
        T   = data[:,-1]
        attr = data[:, :-1]
        res_min = 1e100
        n_res_min = -1
        split_val_min = -1
        calc_buffer = np.zeros(n_T)
        mins = 0
        for n in range(n_attr):
            sorted_ind = np.argsort(data[:,n])
            sorted_T = data[sorted_ind,-1]
            sorted_attr = data[sorted_ind,n]
            u, first_ind = np.unique(sorted_attr,return_index=True)
            # p=1 do all splits p=0 only one split
            p = 1
            stride = int(first_ind.shape[0]*(1-p)+1)
            for ind in first_ind[1::stride]: #iterate splits
                if ind<=mins or (n_T-ind)<=mins:
                    continue
                # get mean
                y_mean0 = (sorted_T[:ind]).mean()
                y_mean1 = (sorted_T[ind:]).mean()
                # subtract next mean
                calc_buffer[:ind] = sorted_T[:ind]-y_mean0
                calc_buffer[ind:] = sorted_T[ind:]-y_mean1
                res = (calc_buffer**2).sum()
                if res<res_min :
                    res_min=res
                    n_res_min = n
                    split_val_min = sorted_attr[ind]
        return res_min, n_res_min, split_val_min
                    
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
            #error_left  = np.mean(np.abs(node.predictby_nodemodel(data[mask_left,:-1])-data[mask_left,-1]))
            error_left  = ((data[mask_left,-1]-node.coeff)**2).mean()
            next_error_left = self.pruneby_abserror(node.left, data[mask_left,:])
            if next_error_left > error_left:
                node.left = None
                print("prune left node at level",node.depth)
            else:
                error_left = next_error_left
        elif np.any(mask_left):
            error_left  = ((data[mask_left,-1]-node.coeff)**2).mean()
        if (node.right != None) and np.any(mask_right):
            error_right  = ((data[mask_right,-1]-node.coeff)**2).mean()
            next_error_right = self.pruneby_abserror(node.right, data[mask_right,:])
            if next_error_right > error_right:
                node.right = None
                print("prune right node at level",node.depth)
            else:
                error_right = next_error_right
        elif np.any(mask_right):
           
            error_right  = ((data[mask_right,-1]-node.coeff)**2).mean()
        if (node.left == None) and (node.right == None):
            node.type = 1
        error_total = error_right+error_left
        if np.any(mask_left) and np.any(mask_right):
            error_total /= 2 # mean of left and right if both contribute
        return error_total
    