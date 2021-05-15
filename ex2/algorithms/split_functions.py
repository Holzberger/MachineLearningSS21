#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 18:19:06 2021

@author: fabian
"""
import numpy as np

def calc_split_residuals(attr, T, return_weighted_std=False):
    res0 = [] # res of left splits
    res1 = [] # res of right splits
    sorted_ind = np.argsort(attr)
    sorted_T = T[sorted_ind]
    sorted_attr = attr[sorted_ind]
    u, first_ind, u_count = np.unique(sorted_attr,return_index=True,
                                      return_counts=True)
    if u.shape[0]<2: # no split possible
        return res0, res1, sorted_attr[first_ind]
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
    #presum2_count_fwd = u_count.cumsum()
    #presum2_count_bwd = (presum2_count_fwd[-1]-presum2_count_fwd)
    binsums2_T = np.zeros_like(first_ind,dtype="float")
    binsums2_T[:-1] = presum2_T[first_ind[1:]-1]-presum2_T[first_ind[:-1]-1]
    binsums2_T[0] = presum2_T[first_ind[1]-1]
    binsums2_T[-1] = presum2_T[-1]-presum2_T[first_ind[-1]-1]
    presum2_means_T_fwd = binsums2_T.cumsum()
    presum2_means_T_bwd = (presum2_means_T_fwd[-1]-presum2_means_T_fwd)
    
    res0 = presum2_means_T_fwd[:-1] - presum_means_T_fwd[:-1]*2*y_mean0 + presum_count_fwd[:-1]*y_mean0**2 
    res1 = presum2_means_T_bwd[:-1] - presum_means_T_bwd[:-1]*2*y_mean1 + presum_count_bwd[:-1]*y_mean1**2
    if return_weighted_std:
        res0 =(res0/presum_count_fwd[:-1])*(presum_count_fwd[:-1]/T.shape[0])
        res1 =(res1/presum_count_bwd[:-1])*(presum_count_bwd[:-1]/T.shape[0])
    
    return res0, res1, sorted_attr[first_ind]

def RMS_residual(data, node):
    n_T, n_attr = data.shape
    n_attr -= 1 
    res_min = 1e100
    n_res_min = -1
    split_val_min = -1
    for n in range(n_attr):
        res0, res1, split_values = calc_split_residuals(data[:,n], data[:,-1])
        if res0!=[]: # continue only if there is a split possible
            res = res0 + res1
            # get minimum residual
            m = np.argmin(res)
            if res[m]<res_min :
                res_min = res[m]
                n_res_min = n
                split_val_min = split_values[m+1]
    return res_min, n_res_min, split_val_min


def SDR(data, node):
    n_T, n_attr = data.shape
    n_attr -= 1 
    T   = data[:,-1]
    sdT = ( (T-T.mean())**2 ).mean()
    SDR_max = -1
    n_SDR_max = -1
    split_SDR_max = -1
    for n in range(n_attr):
        res0, res1, split_values = calc_split_residuals(data[:,n], data[:,-1], return_weighted_std=True)
        if res0!=[]: # continue only if there is a split possible
            SDR_attrs = sdT - res0 - res1
            m = np.argmax(SDR_attrs)
            if SDR_attrs[m] > SDR_max:
                SDR_max = SDR_attrs[m]
                n_SDR_max = n
                split_SDR_max = split_values[m+1]
    return SDR_max, n_SDR_max, split_SDR_max
