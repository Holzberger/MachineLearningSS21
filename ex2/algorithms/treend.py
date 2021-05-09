#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:29:52 2021

@author: fabian
"""
import numpy as np

class Node():
    def __init__(self):
        self.left  = None
        self.right = None
        self.dim   = 0
        self.nval  = 0 # dimension 
        self.val   = 0
        self.type  = 0 # 0 interior, 1 leaf
        self.coeffs = [0, 0]
        
    @classmethod
    def from_vals(self, dim, n_val, val, node_type):
        self.left  = None
        self.right = None
        self.dim   = dim
        self.nval  = n_val # dimension 
        self.val   = val
        self.type   = node_type # 0 interior, 1 leaf
        self.coeffs = [0, 0]

def SDR(data):
    n_T, n_attr = data.shape
    n_attr -= 1 
    T   = data[:,-1]
    sdT = np.std(T)
    attr = data[:, :-1]
    attr_means = np.mean(attr, axis=0) 
    mask_T1 = attr<=attr_means
    SDR_max = -1.0
    for n in range(n_attr):
        T1 = T[mask_T1[:,n]]
        T2 = T[np.logical_not(mask_T1[:,n])]
        SDR_attr = np.abs(sdT - (T1.shape[0]*np.std(T1) + T2.shape[0]*np.std(T2))/n_T)
        if SDR_attr>SDR_max:
            SDR_max = SDR_attr
            n_SDR_max = n
    return data[mask_T1[:,n_SDR_max],:], \
           data[np.logical_not(mask_T1[:,n_SDR_max]),:], \
           attr_means[n_SDR_max], n_SDR_max
            

def split(node, data):
    if data.shape[0] < 20:
        node.type = 1 # leaf
        #node.coeffs = np.polyfit(data[:,:-1], data[:,-1], 1)
        x_coeffs = np.hstack([data[:,:-1], np.ones((data.shape[0],1))])
        node.coeffs = np.linalg.lstsq(x_coeffs, data[:,-1], rcond=None)[0]
    else:
        data_left, data_right, mean, dim_split = SDR(data)
        
        node.type = 0 # interiour
        
        node.left  = Node()
        node.right = Node()
        
        node.val  = mean
        node.nval = dim_split
        
        split(node.left, data_left)
        split(node.right, data_right)
        
    
def create_M5(X):
    x = np.copy(X) # make local copy for tree
    root = Node()
    split(root, x)   
    return root
    
def print_split(node, splits=[]):
    if node.type==0:
        print("{}, ".format(node.val))
        splits.append([node.val, node.nval])
    if node.left != None:
        splits = print_split(node.left, splits)
    if node.right != None:
        splits = print_split(node.right, splits)
    return splits

def predict(node, x):
    if (node.left != None) and (x[:,node.nval]<=node.val):
        return predict(node.left, x)
    elif (node.right != None) and (x[:,node.nval]>node.val):
        return predict(node.right, x)
    else:
        return x.dot(node.coeffs[:-1]) + node.coeffs[-1]