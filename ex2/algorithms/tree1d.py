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


def split(node, data):
    if data.shape[0] < 15:
        node.type = 1 # leaf
        node.coeffs = np.polyfit(data[:,0], data[:,1], 1)
    else:
        node.type = 0 # interiour
        
        node.left  = Node()
        node.right = Node()
        
        node.val = data[:,0].mean(axis=0)
        
        
        split(node.left, data[data[:,0]<=node.val])
        split(node.right, data[data[:,0]>node.val])
        
    
def create_M5(X):
    x = np.copy(X) # make local copy for tree
    root = Node()
    split(root, x)   
    return root
    
    
def print_split(node, splits=[]):
    if node.type==0:
        print("{}, ".format(node.val))
        splits.append(node.val)
    if node.left != None:
        splits = print_split(node.left, splits)
    if node.right != None:
        splits = print_split(node.right, splits)
    return splits

def predict(node, x):
    if (node.left != None) and (x<=node.val):
        return predict(node.left, x)
    elif (node.right != None) and (x>node.val):
        return predict(node.right, x)
    else:
        return node.coeffs[0]*x+node.coeffs[1]
    
    
    
    
    

