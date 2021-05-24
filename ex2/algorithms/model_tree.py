#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:29:52 2021

@author: fabian
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from split_functions import RMS_residual, SDR

class Node():
    root_std = 0
    dims     = 0
    def __init__(self):
        self.left  = None
        self.right = None
        self.dim_split = 0
        self.nval  = 0 
        self.val   = 0
        self.node_std = 0
        self.type  = 0 # 0 interior, 1 leaf
        self.coeffs = np.zeros(self.dims+1,dtype="float64")
        self.fitting_dimensions =[]
        self.error= -1
        self.depth =0
    def predictby_nodemodel(self, x):
        return x.dot(self.coeffs[:-1]) + self.coeffs[-1]
        

class M5regressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 smoothing=True, 
                 pruning=False, 
                 optimize_models=False,
                 incremental_fit=False, 
                 split_function="RMS",
                 n_attr_leaf=4, 
                 max_depth=20, 
                 k=15.0,
                 root_min_std=0.005,
                 prune_set = []):
        self.smoothing = smoothing
        self.n_attr_leaf = n_attr_leaf
        self.max_depth = max_depth
        self.k = k
        self.pruning = pruning
        self.optimize_models = optimize_models
        self.incremental_fit = incremental_fit
        self.split_function = split_function
        self.root_min_std = root_min_std
        self.prune_set = prune_set
        

    
    def fit(self, X, y):
        data = np.hstack((X, y))
        self.root = self.create_M5(data)
        
        if self.pruning:
            if self.prune_set != []:
                self.prune(self.prune_set[0], self.prune_set[1])
            else:
                self.prune(X, y)
                
        return self
    
    def predict(self, X):
        return self.predict_vec(self.root, X, smoothing=self.smoothing)
        
    def create_M5(self, X):
        x = np.copy(X)
        root = Node()
        Node.root_std = np.std(x[:, -1])
        Node.dims = X.shape[1]-1
        self.split(root, x)   
        return root

    
    def split(self, node, data):
        if node.fitting_dimensions ==[]:
            x_coeffs = np.hstack([data[:,:-1], np.ones((data.shape[0],1))])
            node.coeffs = np.linalg.lstsq(x_coeffs, data[:,-1], rcond=None)[0]
            if self.incremental_fit:
                node.fitting_dimensions = np.zeros(node.dims+1,dtype="bool")
                node.fitting_dimensions[-1] = True # offset coeff
            else:
                node.fitting_dimensions = np.ones(node.dims+1,dtype="bool")
        else:
            x_coeffs = np.hstack([data[:,:-1][:,node.fitting_dimensions[:-1]],
                                  np.ones((data.shape[0],1))])
            node.coeffs[node.fitting_dimensions] = np.linalg.lstsq(x_coeffs, data[:,-1], rcond=None)[0]
        node.nval = data.shape[0]
        node.node_std = np.std(data[:,-1])
        if (node.nval < self.n_attr_leaf) or\
            (node.node_std<node.root_std*self.root_min_std) or\
            (node.depth>self.max_depth) or\
            (np.max(data[:,:-1],axis=0)-np.min(data[:,:-1],axis=0)<1e-12).all():
            node.type = 1 # leaf
        else:
            #print("calc split of {} points".format(data.shape[0]))
            if self.split_function=="RMS":
                node.error, node.dim_split, node.val = RMS_residual(data, node)
            elif self.split_function=="SDR":
                node.error, node.dim_split, node.val = SDR(data, node)
                
            if node.dim_split==-1: # split not found
                node.type = 1 # leaf
            else:
                #print("dim, split",node.dim_split, node.val )
                
                mask_left  = data[:,node.dim_split]<node.val
                mask_right = np.logical_not(mask_left)
                data_left = data[mask_left,:]
                data_right = data[mask_right,:]
                
                node.type = 0 # interiour
                node.left  = Node()
                node.right = Node()
                next_fitting_dimsions = np.copy(node.fitting_dimensions)
                next_fitting_dimsions[node.dim_split] = True
                node.left.fitting_dimensions = next_fitting_dimsions
                node.right.fitting_dimensions = next_fitting_dimsions
                node.left.depth = node.depth+1
                node.right.depth = node.depth+1
                self.split(node.left, data_left)
                self.split(node.right, data_right)
        
            
    def predict_vec(self, node, x, smoothing=False):
        y = np.zeros((x.shape[0]))
        mask_left = x[:,node.dim_split] < node.val
        mask_right = np.logical_not(mask_left)
        
        if (node.left != None) and np.any(mask_left):
            if self.smoothing:
                y[mask_left] = (node.left.nval*self.predict_vec(node.left, x[mask_left,:])+\
                                self.k*node.predictby_nodemodel(x[mask_left,:]))/(node.left.nval+self.k)
            else:
                y[mask_left] = self.predict_vec(node.left, x[mask_left,:])
        elif np.any(mask_left):
            y[mask_left] = node.predictby_nodemodel(x[mask_left,:])
            
        if (node.right != None) and np.any(mask_right):
            if self.smoothing:
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
        mask_left = data[:,node.dim_split]<node.val
        mask_right = np.logical_not(mask_left)
        error_left = 0.0
        error_right = 0.0
        if (node.left != None) and np.any(mask_left):
            #error_left  = np.mean(np.abs(node.predictby_nodemodel(data[mask_left,:-1])-data[mask_left,-1]))
            error_left  = self.find_best_coeff(node, data[mask_left,:])
            next_error_left = self.pruneby_abserror(node.left, data[mask_left,:])
            if next_error_left > error_left:
                node.left = None
                #print("prune right node at level",node.depth)
            else:
                error_left = next_error_left
        elif np.any(mask_left):
            #error_left  = np.mean(np.abs(node.predictby_nodemodel(data[mask_left,:-1])-data[mask_left,-1]))
            error_left  = self.find_best_coeff(node, data[mask_left,:])
        if (node.right != None) and np.any(mask_right):
            #error_right = np.mean(np.abs(node.predictby_nodemodel(data[mask_right,:-1])-data[mask_right,-1]))
            error_right  = self.find_best_coeff(node, data[mask_right,:])
            next_error_right = self.pruneby_abserror(node.right, data[mask_right,:])
            if next_error_right > error_right:
                node.right = None
                #print("prune left node at level",node.depth)
            else:
                error_right = next_error_right
        elif np.any(mask_right):
            #error_right = np.mean(np.abs(node.predictby_nodemodel(data[mask_right,:-1])-data[mask_right,-1]))
            error_right  = self.find_best_coeff(node, data[mask_right,:])
        if (node.left == None) and (node.right == None):
            node.type = 1
        error_total = error_right+error_left
        if np.any(mask_left) and np.any(mask_right):
            error_total /= 2 # mean of left and right if both contribute
        return error_total
        
    def find_best_coeff(self, node, data):
        buffer = 0.0
        min_error = ((node.predictby_nodemodel(data[:,:-1])-data[:,-1])**2).sum()
        if self.optimize_models:
            for i, coeff in enumerate(node.coeffs):
                buffer = coeff
                node.coeffs[i] = 0.0
                error = ((node.predictby_nodemodel(data[:,:-1])-data[:,-1])**2).sum()
                if min_error > error:
                    min_error=error
                else:
                    node.coeffs[i] = buffer
        return min_error
        
        
        
    