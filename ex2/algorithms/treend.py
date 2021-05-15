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
    def __init__(self, smoothing=True, pruning=False, 
                 optimize_models=False,incremental_fit=False, n_attr_leaf=4, 
                 max_depth=20, k=15.0):
        self.smoothing = smoothing
        self.n_attr_leaf = n_attr_leaf
        self.max_depth = max_depth
        self.k = k
        self.pruning = pruning
        self.optimize_models = optimize_models
        self.incremental_fit = incremental_fit
        
    
    def fit(self, X, y):
        data = np.hstack((X, y))
        self.root = self.create_M5(data)
        if self.pruning:
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
    
    def rRMS(self, data, node):
        n_T, n_attr = data.shape
        n_attr -= 1 
        T   = data[:,-1]
        attr = data[:, :-1]
        res_min = 1e100
        n_res_min = -1
        split_val_min = -1
        calc_buffer = np.zeros(n_T)
        for n in range(n_attr):
            sorted_ind = np.argsort(data[:,n])
            sorted_T = data[sorted_ind,-1]
            sorted_attr = data[sorted_ind,n]
            u, first_ind, u_count = np.unique(sorted_attr,return_index=True,return_counts=True)
            
            if u.shape[0]<2: # no split possible
                continue
            
            # sum of target y in the splits
            presum_T = sorted_T.cumsum()
            presum_count_fwd = u_count.cumsum()
            presum_count_bwd = (presum_count_fwd[-1]-presum_count_fwd)
            means_T = np.zeros_like(first_ind,dtype="float")
            means_T[:-1] = presum_T[first_ind[1:]-1]-presum_T[first_ind[:-1]-1]
            means_T[0] = presum_T[first_ind[1]-1]
            means_T[-1] = presum_T[-1]-presum_T[first_ind[-1]-1]
            presum_means_T_fwd = means_T.cumsum()
            presum_means_T_bwd = (presum_means_T_fwd[-1]-presum_means_T_fwd)
            # mean values per split
            y_mean0 = presum_means_T_fwd[:-1]/presum_count_fwd[:-1]
            y_mean1 = presum_means_T_bwd[:-1]/presum_count_bwd[:-1]
            # sum of target y**2 in the splits 
            presum2_T = (sorted_T**2).cumsum()
            presum2_count_fwd = u_count.cumsum()
            presum2_count_bwd = (presum2_count_fwd[-1]-presum2_count_fwd)
            binsums2_T = np.zeros_like(first_ind,dtype="float")
            binsums2_T[:-1] = presum2_T[first_ind[1:]-1]-presum2_T[first_ind[:-1]-1]
            binsums2_T[0] = presum2_T[first_ind[1]-1]
            binsums2_T[-1] = presum2_T[-1]-presum2_T[first_ind[-1]-1]
            presum2_means_T_fwd = binsums2_T.cumsum()
            presum2_means_T_bwd = (presum2_means_T_fwd[-1]-presum2_means_T_fwd)
            # residuals in all splits for this attribute
            res = presum2_means_T_fwd[:-1] + presum2_means_T_bwd[:-1] -\
                  presum_means_T_fwd[:-1]*2*y_mean0 - presum_means_T_bwd[:-1]*2*y_mean1 +\
                  presum_count_fwd[:-1]*y_mean0**2 + presum_count_bwd[:-1]*y_mean1**2
            # minimum residual
            m = np.argmin(res)
            
            if res[m]<res_min :
                res_min = res[m]
                n_res_min = n
                split_val_min = sorted_attr[first_ind[m+1]]
                
        mask_left  = data[:,n_res_min]<split_val_min
        mask_right = np.logical_not(mask_left)
        return data[mask_left,:], \
                data[mask_right,:], \
                split_val_min, \
                n_res_min

        return res_min, n_res_min, split_val_min
    
    
    def SDR(self, data, node):
        n_T, n_attr = data.shape
        n_attr -= 1 
        T   = data[:,-1]
        sdT = T.std()
        attr = data[:, :-1]
        attr_means = np.mean(attr, axis=0) 
        mask_T1 = attr <= attr_means
        SDR_max = -1
        n_SDR_max = -1
        split_SDR_max = -1
        mins = 0
        
        for n in range(n_attr):
            sorted_ind = np.argsort(data[:,n])
            sorted_T = data[sorted_ind,-1]
            sorted_attr = data[sorted_ind,n]
            u, first_ind, u_count = np.unique(sorted_attr,return_index=True,return_counts=True)
            
            
            presum_T = sorted_T.cumsum()

            presum_count_fwd = u_count.cumsum()
            
            presum_count_bwd = (presum_count_fwd[-1]-presum_count_fwd)
            
            means_T = np.zeros_like(first_ind)
            means_T[:-1] = presum_T[first_ind[1:]-1]-presum_T[first_ind[:-1]-1]
            means_T[0] = presum_T[first_ind[1]-1]
            means_T[-1] = presum_T[-1]-presum_T[first_ind[-1]-1]
            
            presum_means_T_fwd = means_T.cumsum()
            presum_means_T_bwd = (presum_means_T_fwd[-1]-presum_means_T_fwd)
            
            means0 = presum_means_T_fwd[:-1]/presum_count_fwd[:-1]
            means1 = presum_means_T_bwd[:-1]/presum_count_bwd[:-1]
            
            
            j=0
            # p=1 do all splits p=0 only one split
            p = 1
            stride = int(first_ind.shape[0]*(1-p)+1)
            for ind in first_ind[1::stride]: #iterate splits
                #if ind<=mins or (n_T-ind)<=mins:
                    #continue
                # get mean
                buff0 = (sorted_T[:ind])
                buff1 = (sorted_T[ind:])
                y_std0   = buff0.std()
                y_std1   = buff1.std()
                SDR_attr = sdT - (ind*y_std0 + (n_T-ind)*y_std1)/n_T
                
                m0 = means0[j]
                m0c = (sorted_T[:ind]).mean()
                m1 = means1[j]
                m1c = (sorted_T[ind:]).mean()
                
                
                j+=1
                if j==5483:
                    a=1
                if SDR_attr > SDR_max:
                    SDR_max = SDR_attr
                    n_SDR_max = n
                    split_SDR_max = sorted_attr[ind]
                
        mask_left  = data[:,n_SDR_max]<split_SDR_max
        mask_right = np.logical_not(mask_left)
        return data[mask_left,:], \
                data[mask_right,:], \
                split_SDR_max, \
                n_SDR_max
               
        
    
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
            (node.node_std<node.root_std*0.005) or\
            (node.depth>self.max_depth) or\
            (np.max(data[:,:-1],axis=0)-np.min(data[:,:-1],axis=0)<1e-12).all():
            node.type = 1 # leaf
        else:
            print("calc SDR of {} points".format(data.shape[0]))
            data_left, data_right, mean, dim_split = self.rRMS(data, node)
            if dim_split==-1: # split not found
                node.type = 1 # leaf
            else:
                print("dim, split",dim_split, mean )
                node.type = 0 # interiour
                node.left  = Node()
                node.right = Node()
                next_fitting_dimsions = np.copy(node.fitting_dimensions)
                next_fitting_dimsions[dim_split] = True
                node.left.fitting_dimensions = next_fitting_dimsions
                node.right.fitting_dimensions = next_fitting_dimsions
                node.left.depth = node.depth+1
                node.right.depth = node.depth+1
                node.val  = mean
                node.dim_split = dim_split
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
                print("prune right node at level",node.depth)
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
                print("prune left node at level",node.depth)
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
        
        
        
    