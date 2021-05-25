#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 19:36:28 2021

@author: fabian
"""

import numpy as np
import pandas as pd

import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor

from model_tree import *
from reg_tree import *
from sklearn.linear_model import *
from lineartree import LinearTreeClassifier, LinearTreeRegressor

#%%

link ="https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"

df = pd.read_csv(link)
#%%
print("Missing vals:")
print(df.isnull().any())

# convert to date datatype
df['date_time'] = pd.to_datetime(df['date_time'])

# extract year, day and hour from date
df['year'] = [d.year for d in df['date_time']]
df['day_year'] = [d.dayofyear for d in df['date_time']]
df['day_week'] = [d.dayofweek for d in df['date_time']]
df['hour'] = [d.hour for d in df['date_time']]
df['month'] = [d.month for d in df['date_time']]

# date is not needed anymore
df=df.drop(columns=['date_time'])

#%% replace zero kelvin samples by mean
error_ind = np.where(df['temp']<1.0)[0]
print("Temperature measurement errors: ", error_ind.shape[0] )
df['temp'][error_ind] = np.mean(df['temp'][df['temp']>1.0])
#%% make holiday binary
df['holiday'][df['holiday']!="None"] = 1
df['holiday'][df['holiday']=="None"] = 0


#%% encode remaining cathegorical attrs as one hot
df = pd.get_dummies(df, columns=["weather_main", "weather_description"], prefix=["weather_main", "weather_description"])

#%% create test train split
x=df.drop(columns=["traffic_volume"])
y=df["traffic_volume"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
x_train = np.array(x_train).astype("float")
x_test = np.array(x_test).astype("float")
y_train = np.array(y_train).astype("float")
y_test = np.array(y_test).astype("float")
#%%save results here
def get_res_dict(model_names):
    k_fold_res = {}
    for name in model_names:
        k_fold_res[name] = [[],[],[]]
    return k_fold_res
#%% functions to fit models  
def evaluate_models(res_dict, X_train, Y_train, X_test, Y_test, n_attr_leaf= 4, pruning= False, smoothing= False, 
                    optimized_model= False, incremental_fit = False, k=15, 
                    max_depth = 99, split_function="RMS", min_samples_split= 6, skip_compares=True):
    print("evaluate m5")
    start = time.time()
    reg = M5regressor(smoothing=smoothing, n_attr_leaf= n_attr_leaf , max_depth=max_depth, 
                      k=k,pruning=pruning,optimize_models=optimized_model,incremental_fit=incremental_fit).fit(X_train, Y_train[:,None])
    res_dict["m5"][0].append(r2_score(Y_test, reg.predict(X_test)))
    res_dict["m5"][1].append(mean_absolute_error(Y_test, reg.predict(X_test)))
    res_dict["m5"][2].append(time.time()-start)
    #Constant regressor tree own
    print("evaluate const")
    start = time.time()
    reg2 = Const_regressor(n_attr_leaf = n_attr_leaf , max_depth=max_depth, smoothing=smoothing, k=k, split_function=split_function, pruning=pruning).fit(X_train, Y_train[:,None])
    res_dict["const"][0].append(r2_score(Y_test, reg2.predict(X_test)))
    res_dict["const"][1].append(mean_absolute_error(Y_test, reg2.predict(X_test)))
    res_dict["const"][2].append(time.time()-start)
    if not skip_compares:
        #Constant regressor tree SKlearn
        print("evaluate const_sk")
        start = time.time()
        const_regressor_sklearn = tree.DecisionTreeRegressor(min_samples_leaf=n_attr_leaf , max_depth=max_depth, random_state=42).fit(X_train, Y_train)
        res_dict["const_sk"][0].append(r2_score(Y_test, const_regressor_sklearn.predict(X_test)))
        res_dict["const_sk"][1].append(mean_absolute_error(Y_test, const_regressor_sklearn.predict(X_test)))
        res_dict["const_sk"][2].append(time.time()-start)
        #Linear regression sklearn
        print("evaluate lin")
        start = time.time()
        linear_regressor = LinearRegression().fit(X_train, Y_train)
        res_dict["lin"][0].append(r2_score(Y_test, linear_regressor.predict(X_test)))
        res_dict["lin"][1].append(mean_absolute_error(Y_test, linear_regressor.predict(X_test)))
        res_dict["const_sk"][2].append(time.time()-start)
        # Random forrest regressor
        print("evaluate forrest")
        start = time.time()
        random_forest_regressor = RandomForestRegressor(max_depth=max_depth, min_samples_split=n_attr_leaf , random_state=42).fit(X_train, Y_train)
        res_dict["forrest"][0].append(r2_score(Y_test, random_forest_regressor.predict(X_test)))
        res_dict["forrest"][1].append(mean_absolute_error(Y_test, random_forest_regressor.predict(X_test)))
        res_dict["forrest"][2].append(time.time()-start)
        #Extra tree
        print("evaluate extra")
        start = time.time()
        extra_tree_regressor = ExtraTreeRegressor(max_depth=max_depth, min_samples_split= n_attr_leaf , random_state=42).fit(X_train, Y_train)
        res_dict["extra"][0].append(r2_score(Y_test, extra_tree_regressor .predict(X_test)))
        res_dict["extra"][1].append(mean_absolute_error(Y_test, extra_tree_regressor .predict(X_test)))
        res_dict["extra"][2].append(time.time()-start)
        # linear tree (model tree from Gitbhub) 
        print("evaluate lin_tree")
        start = time.time()
        linear_tree_regressor = LinearTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, base_estimator=LinearRegression()).fit(X_train, Y_train) 
        res_dict["lin_tree"][0].append(r2_score(Y_test, linear_tree_regressor.predict(X_test)))
        res_dict["lin_tree"][1].append(mean_absolute_error(Y_test, linear_tree_regressor.predict(X_test)))
        res_dict["lin_tree"][2].append(time.time()-start)
def evaluate_models_k(res_dict, X_train, Y_train, X_test, Y_test, n_attr_leaf= 4, pruning= False, smoothing= True, 
                    optimized_model= False, incremental_fit = False, ks=[], 
                    max_depth = 99, split_function="RMS"):
    reg = M5regressor(smoothing=smoothing, n_attr_leaf= n_attr_leaf , max_depth=max_depth, 
                      k=1,pruning=pruning,optimize_models=optimized_model,incremental_fit=incremental_fit).fit(X_train, Y_train[:,None])
    reg2 = Const_regressor(n_attr_leaf = n_attr_leaf , max_depth=max_depth, smoothing=smoothing, k=1, split_function=split_function, pruning=pruning).fit(X_train, Y_train[:,None])
    for n, k in enumerate(ks):
        reg.k = k
        reg2.k = k
        res_dict["m5"][0].append(r2_score(Y_test, reg.predict(X_test)))
        res_dict["m5"][1].append(mean_absolute_error(Y_test, reg.predict(X_test)))
        res_dict["const"][0].append(r2_score(Y_test, reg2.predict(X_test)))
        res_dict["const"][1].append(mean_absolute_error(Y_test, reg2.predict(X_test)))
        
def evaluate_models_prune(res_dict, X_train, Y_train, X_test, Y_test, n_attr_leaf= 4, pruning= False, smoothing= False, 
                    optimized_model= False, incremental_fit = False, 
                    max_depth = 99, split_function="RMS"):
    reg = M5regressor(smoothing=smoothing, n_attr_leaf= n_attr_leaf , max_depth=max_depth, 
                      k=1,pruning=pruning,optimize_models=optimized_model,incremental_fit=incremental_fit).fit(X_train, Y_train[:,None])
    reg2 = Const_regressor(n_attr_leaf = n_attr_leaf , max_depth=max_depth, smoothing=smoothing, k=1, split_function=split_function, pruning=pruning).fit(X_train, Y_train[:,None])
    res_dict["m5"][0].append(r2_score(Y_test, reg.predict(X_test)))
    res_dict["m5"][1].append(mean_absolute_error(Y_test, reg.predict(X_test)))
    res_dict["const"][0].append(r2_score(Y_test, reg2.predict(X_test)))
    res_dict["const"][1].append(mean_absolute_error(Y_test, reg2.predict(X_test)))
    reg.prune(X_test, Y_test[:,None])
    reg2.prune(X_test, Y_test[:,None])
    res_dict["m5"][0].append(r2_score(Y_test, reg.predict(X_test)))
    res_dict["m5"][1].append(mean_absolute_error(Y_test, reg.predict(X_test)))
    res_dict["const"][0].append(r2_score(Y_test, reg2.predict(X_test)))
    res_dict["const"][1].append(mean_absolute_error(Y_test, reg2.predict(X_test)))
    
def tune_models(res_dict, X_train, Y_train, X_test, Y_test, n_attr_leaf= [4,4], pruning= False, smoothing= [False,False], 
                    optimized_model= False, incremental_fit = False, k=[15,15], 
                    max_depth = [99,99], split_function="RMS"):
    reg = M5regressor(smoothing=smoothing[1], n_attr_leaf= n_attr_leaf[1] , max_depth=max_depth[1], 
                      k=k[1],pruning=pruning,optimize_models=optimized_model,incremental_fit=incremental_fit).fit(X_train, Y_train[:,None])
    reg2 = Const_regressor(n_attr_leaf = n_attr_leaf[0] , max_depth=max_depth[0], smoothing=smoothing[0], k=k[0], split_function=split_function, pruning=pruning).fit(X_train, Y_train[:,None])
    if pruning[1]:
        reg.prune(X_test, Y_test[:,None])
    if pruning[0]:
        reg2.prune(X_test, Y_test[:,None])
    res_dict["m5"][0].append(r2_score(Y_test, reg.predict(X_test)))
    res_dict["m5"][1].append(mean_absolute_error(Y_test, reg.predict(X_test)))
    res_dict["const"][0].append(r2_score(Y_test, reg2.predict(X_test)))
    res_dict["const"][1].append(mean_absolute_error(Y_test, reg2.predict(X_test)))
        
#%%
n_splits=5
#%%
def own_tree_plots(results, parameter, title1="", ylab1="R2 score", xlab1="",
              title2="", ylab2="MAE score", xlab2="",
              x1scale="linear",y1scale="linear",x2scale="linear",y2scale="linear",n_splits=2):
    fig, ax = plt.subplots(1,2, figsize = (10,6))

    ax[0].set_title(title1,fontsize=12,fontweight='bold')
    ax[0].set_ylabel(ylab1,fontsize=12,fontweight='bold')
    ax[0].set_xlabel(xlab1,fontsize=12,fontweight='bold')
    ax[0].plot(parameter, np.array(results["m5"][0]).reshape(-1,n_splits).mean(axis=1),label="m5 tree",marker="x",linewidth=2)
    ax[0].plot(parameter, np.array(results["const"][0]).reshape(-1,n_splits).mean(axis=1),label="const tree",marker="x",linewidth=2)
    ax[0].set_ylim(top=4)
    ax[0].grid(True, which="both", ls="-")
    ax[0].legend(loc="best")
    ax[0].set_yscale(y1scale)
    ax[0].set_xscale(x1scale)
    
    ax[1].set_title(title2, fontsize=12,fontweight='bold')
    ax[1].set_ylabel(ylab2, fontsize=12,fontweight='bold')
    ax[1].set_xlabel(xlab2, fontsize=12,fontweight='bold')
    ax[1].plot(parameter, np.array(results["m5"][1]).reshape(-1,n_splits).mean(axis=1),label="m5 tree",marker="x",linewidth=2)
    ax[1].plot(parameter, np.array(results["const"][1]).reshape(-1,n_splits).mean(axis=1),label="const tree",marker="x",linewidth=2)
    ax[1].grid(True, which="both", ls="-")
    ax[1].legend(loc="best")
    ax[1].set_yscale(y2scale)
    ax[1].set_xscale(x2scale)
#%%

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

model_names = ["m5","const"]
depth_reults = get_res_dict(model_names)
depths = [1,2,4,6,8,10,15,20,22,24]

for depth in depths:
    for train_index, test_index in kf.split(x_train):
        X_train, Y_train, X_test, Y_test = x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index]
        evaluate_models(depth_reults, X_train, Y_train, X_test, Y_test,max_depth=depth)

#%%
own_tree_plots(depth_reults, depths, xlab1="max depth", xlab2="max depth",y1scale="symlog",n_splits=n_splits)
plt.savefig("./traffic_vol_maxdepth.svg")
#%%
model_names = ["m5","const"]
min_samples_reults = get_res_dict(model_names)
min_samples = [4,10,50,100,300,1000,2000,5000]

for min_sample in min_samples:
    for train_index, test_index in kf.split(x_train):
        X_train, Y_train, X_test, Y_test = x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index]
        evaluate_models(min_samples_reults, X_train, Y_train, X_test, Y_test,n_attr_leaf=min_sample)
#%%
own_tree_plots(min_samples_reults, min_samples, xlab1="min samples per node", xlab2="min samples per node",
               x1scale="log",y1scale="symlog",x2scale="log",n_splits=n_splits)
plt.savefig("./traffic_vol_minsamp.svg")
#%%
model_names = ["m5","const"]

ks = [0.1,1,3,10,20,50,100,150,200,400]
ks_results = get_res_dict(model_names)


for train_index, test_index in kf.split(x_train):
    X_train, Y_train, X_test, Y_test = x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index]
    evaluate_models_k(ks_results, X_train, Y_train, X_test, Y_test,ks=ks, smoothing=True)
    
ks_results["const"][0] =np.array(ks_results["const"][0]).reshape(-1,len(ks)).T.flatten()
ks_results["const"][1] =np.array(ks_results["const"][1]).reshape(-1,len(ks)).T.flatten()
ks_results["m5"][0] =np.array(ks_results["m5"][0]).reshape(-1,len(ks)).T.flatten()
ks_results["m5"][1] =np.array(ks_results["m5"][1]).reshape(-1,len(ks)).T.flatten()
#%%
own_tree_plots(ks_results, ks, xlab1="k", xlab2="k",
               y1scale="symlog",x1scale="log",x2scale="log",n_splits=n_splits)
plt.savefig("./traffic_vol_smooting.svg")
#%%
model_names = ["m5","const"]

pruned = [0,1]
prune_results = get_res_dict(model_names)


for train_index, test_index in kf.split(x_train):
    X_train, Y_train, X_test, Y_test = x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index]
    evaluate_models_prune(prune_results, X_train, Y_train, X_test, Y_test)
    #%%
prune_results["const"][0] =np.array(prune_results["const"][0]).reshape(-1,len(pruned)).T.flatten()
prune_results["const"][1] =np.array(prune_results["const"][1]).reshape(-1,len(pruned)).T.flatten()
prune_results["m5"][0] =np.array(prune_results["m5"][0]).reshape(-1,len(pruned)).T.flatten()
prune_results["m5"][1] =np.array(prune_results["m5"][1]).reshape(-1,len(pruned)).T.flatten()
#%%
own_tree_plots(prune_results, pruned, xlab1="pruned", xlab2="pruned",
               y1scale="linear",n_splits=n_splits)
plt.savefig("./traffic_vol_pruning.svg")
#%% do some manual tuning for best parameters
model_names = ["m5","const"]
tune_results = get_res_dict(model_names)

for train_index, test_index in kf.split(x_train):
    print("evaluate split")
    X_train, Y_train, X_test, Y_test = x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index]
    tune_models(tune_results, X_train, Y_train, X_test, Y_test,n_attr_leaf= [50,4], 
                          pruning= [False,True], smoothing= [False,False], optimized_model= False, 
                          incremental_fit = False, k=[100,800], max_depth = [10,6], split_function="RMS")
#%%
print("const r2:", np.mean(tune_results["const"][0]))
print("const MAE:", np.mean(tune_results["const"][1]))
print("m5 r2:", np.mean(tune_results["m5"][0]))
print("m5 MAE:", np.mean(tune_results["m5"][1]))


#%% evaluate final results
def get_final_results(res_dict, X_train, Y_train, X_test, Y_test, x_test, y_test):
    print("evaluate m5")
    start = time.time()
    reg = M5regressor(smoothing=False, n_attr_leaf= 4 , max_depth=6, k=0,pruning=False,optimize_models=False,incremental_fit=False).fit(X_train, Y_train[:,None])
    reg.prune(X_test, Y_test[:,None]) # prune manually
    res_dict["m5"][0].append(r2_score(y_test, reg.predict(x_test)))
    res_dict["m5"][1].append(mean_absolute_error(y_test, reg.predict(x_test)))
    res_dict["m5"][2].append(time.time()-start)
    #Constant regressor tree own
    print("evaluate const")
    start = time.time()
    reg2 = Const_regressor(n_attr_leaf = 50 , max_depth=10, smoothing=False, k=0, pruning=False).fit(X_train, Y_train[:,None])
    res_dict["const"][0].append(r2_score(y_test, reg2.predict(x_test)))
    res_dict["const"][1].append(mean_absolute_error(y_test, reg2.predict(x_test)))
    res_dict["const"][2].append(time.time()-start)
    #Constant regressor tree SKlearn
    print("evaluate const_sk")
    start = time.time()
    const_regressor_sklearn = tree.DecisionTreeRegressor(min_samples_leaf=50 , max_depth=10, random_state=42).fit(X_train, Y_train)
    res_dict["const_sk"][0].append(r2_score(y_test, const_regressor_sklearn.predict(x_test)))
    res_dict["const_sk"][1].append(mean_absolute_error(y_test, const_regressor_sklearn.predict(x_test)))
    res_dict["const_sk"][2].append(time.time()-start)
    #Linear regression sklearn
    print("evaluate lin")
    start = time.time()
    linear_regressor = LinearRegression().fit(X_train, Y_train)
    res_dict["lin"][0].append(r2_score(y_test, linear_regressor.predict(x_test)))
    res_dict["lin"][1].append(mean_absolute_error(y_test, linear_regressor.predict(x_test)))
    res_dict["lin"][2].append(time.time()-start)
    # Random forrest regressor
    print("evaluate forrest")
    start = time.time()
    random_forest_regressor = RandomForestRegressor( random_state=42).fit(X_train, Y_train)
    res_dict["forrest"][0].append(r2_score(y_test, random_forest_regressor.predict(x_test)))
    res_dict["forrest"][1].append(mean_absolute_error(y_test, random_forest_regressor.predict(x_test)))
    res_dict["forrest"][2].append(time.time()-start)
    #Extra tree
    print("evaluate extra")
    start = time.time()
    extra_tree_regressor = ExtraTreeRegressor(random_state=42).fit(X_train, Y_train)
    res_dict["extra"][0].append(r2_score(y_test, extra_tree_regressor .predict(x_test)))
    res_dict["extra"][1].append(mean_absolute_error(y_test, extra_tree_regressor .predict(x_test)))
    res_dict["extra"][2].append(time.time()-start)
    # linear tree (model tree from Gitbhub) 
    print("evaluate lin_tree")
    start = time.time()
    linear_tree_regressor = LinearTreeRegressor(max_depth=6, min_samples_split=6, base_estimator=LinearRegression()).fit(X_train, Y_train) 
    res_dict["lin_tree"][0].append(r2_score(y_test, linear_tree_regressor.predict(x_test)))
    res_dict["lin_tree"][1].append(mean_absolute_error(y_test, linear_tree_regressor.predict(x_test)))
    res_dict["lin_tree"][2].append(time.time()-start)
#%%
model_names = ["m5","const", "const_sk","lin","forrest","extra","lin_tree"]
final_results = get_res_dict(model_names)

for train_index, test_index in kf.split(x_train):
    X_train, Y_train, X_test, Y_test = x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index]
    get_final_results(final_results, X_train, Y_train, X_test, Y_test, x_test, y_test)
#%%
for name in model_names:
    final_results[name][0] = np.array(final_results[name][0]).mean()
    final_results[name][1] = np.array(final_results[name][1]).mean()
    final_results[name][2] = np.array(final_results[name][2]).mean()
#%%
objects = ('M5', 'Cons', 'Cons_sklearn', 'linear tree', 'linear', 'forrest', 'Extra tree')
y_pos = np.arange(len(objects))
r2Score = [final_results["m5"][0], 
           final_results["const"][0], 
           final_results["const_sk"][0], 
           final_results["lin_tree"][0], 
           final_results["lin"][0], 
           final_results["forrest"][0], 
           final_results["extra"][0]]

plt.bar(y_pos, r2Score, align='center', alpha=0.5, color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('R2 score')
plt.ylim([-0.1,1])
plt.title('R2 score per model')
plt.xticks(rotation = 20, ha="right")
plt.savefig("./traffic_vol_r2_perf.jpg")
#%%
y_pos = np.arange(len(objects))
MAE_score = [final_results["m5"][1], 
           final_results["const"][1], 
           final_results["const_sk"][1], 
           final_results["lin_tree"][1], 
           final_results["lin"][1], 
           final_results["forrest"][1], 
           final_results["extra"][1]]

plt.bar(y_pos, MAE_score, align='center', alpha=0.5, color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('MAE score')
#plt.ylim(top=2000)
plt.yscale('log')
plt.title('MAE score per model')
plt.xticks(rotation = 20, ha="right")
plt.savefig("./traffic_vol_mae_perf.jpg")
#%%
y_pos = np.arange(len(objects))
MAE_score = [final_results["m5"][2], 
           final_results["const"][2], 
           final_results["const_sk"][2], 
           final_results["lin_tree"][2], 
           final_results["lin"][2], 
           final_results["forrest"][2], 
           final_results["extra"][2]]

plt.bar(y_pos, MAE_score, align='center', alpha=0.5, color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Time[s]')
#plt.ylim(top=2000)
#plt.yscale('log')
plt.title('Execution Time per Model')
plt.xticks(rotation = 20, ha="right")
plt.savefig("./traffic_vol_exectime_perf.jpg")
