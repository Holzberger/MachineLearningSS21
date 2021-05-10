#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:29:52 2021

@author: fabian
"""
import numpy as np

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
    def predictby_nodemodel(self, x):
        return x.dot(self.coeffs[:-1]) + self.coeffs[-1]
        


def SDR(data, node):
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
            

def split(node, data):
    # setup linear model in all nodes
    x_coeffs = np.hstack([data[:,:-1], np.ones((data.shape[0],1))])
    node.coeffs = np.linalg.lstsq(x_coeffs, data[:,-1], rcond=None)[0]
    node.nval = data.shape[0]
    node.node_std = np.std(data[:,-1])
    if (node.nval < 4) or (node.node_std<node.root_std*0.05):
        node.type = 1 # leaf
    else:
        data_left, data_right, mean, dim_split = SDR(data, node)
        node.type = 0 # interiour
        node.left  = Node()
        node.right = Node()
        node.val  = mean
        node.dim_split = dim_split
        split(node.left, data_left)
        split(node.right, data_right)
        
    
def create_M5(X):
    x = np.copy(X)
    root = Node()
    root.root_std = np.std(x[:, -1])
    split(root, x)   
    return root
    
def print_split(node, splits=[]):
    if node.type==0:
        print("{}, {} ".format(node.val, node.dim_split))
        splits.append([node.val, node.dim_split])
    if node.left != None:
        splits = print_split(node.left, splits)
    if node.right != None:
        splits = print_split(node.right, splits)
    return splits


def predict(node, x, smoothing=False):
    # assume for this function x is only one miltidimenaional sample x=[x1,x2,...,xn]
    k = 15.0 #smoothing coefficient
    if (node.left != None) and (x[:,node.dim_split]<=node.val):
        if smoothing:
            return (node.nval*predict(node.left, x, smoothing) +\
                k*node.predictby_nodemodel(x))/(node.nval + k)
        else:
            return predict(node.left, x, smoothing)
    elif (node.right != None) and (x[:,node.dim_split]>node.val):
        if smoothing:
            return (node.nval*predict(node.right, x, smoothing) +\
                k*node.predictby_nodemodel(x))/(node.nval + k)
        else:
            return predict(node.right, x, smoothing)
    else:
        return node.predictby_nodemodel(x)
    
def prune(node,data):
    # assume here data are multiple samples (the test set)
    gen_abs_error(node, data)


def gen_abs_error(node, data):
    mask_left = data[:,node.dim_split]<=node.val
    mask_right = np.logical_not(mask_left)
    error_left = 0.0
    error_right = 0.0
    if (node.left != None) and np.any(mask_left):
        error_left  = np.max(np.abs(node.predictby_nodemodel(data[mask_left,:-1])-data[mask_left,-1]))
        next_error_left = gen_abs_error(node.left, data[mask_left,:])
        if next_error_left > error_left:
            node.left = None
        else:
            error_left = next_error_left
    if (node.right != None) and np.any(mask_right):
        error_right = np.max(np.abs(node.predictby_nodemodel(data[mask_right,:-1])-data[mask_right,-1]))
        next_error_right = gen_abs_error(node.right, data[mask_right,:])
        if next_error_right > error_right:
            node.right = None
        else:
            error_right = next_error_right
    if (node.left == None) and (node.right == None):
        node.type = 1
        # if we have a leaf the error is calculated directly from the model
        error_left = np.max(np.abs(node.predictby_nodemodel(data[:,:-1])-data[:,-1]))
        error_right = error_left
        
    return (error_right+error_left)/2.0
    
        
        
        
        
        
        
        
    