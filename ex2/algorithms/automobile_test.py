#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:55:45 2021

@author: fabian
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from model_tree import *
from reg_tree import *

from sklearn.linear_model import *
from lineartree import LinearTreeClassifier, LinearTreeRegressor

df = pd.read_csv('../datasets/Automobile_data.csv')

#%% handle missing values
# Converting price column type to numeric & Dropping '?' entries from price column 
df['price'] = df['price'].replace('?',np.nan)
df['price'] = pd.to_numeric(df['price'])
df = df[df['price'].notna()]

# Dropping '?' from num-of-doors column 
df['num-of-doors'] = df['num-of-doors'].replace('?',np.nan)
df = df[df['num-of-doors'].notna()]

# replace missing vals with mean
num_col = ['normalized-losses', 'bore',  'stroke', 'horsepower', 'peak-rpm']
for col in num_col:
    df[col] = df[col].replace('?', np.nan)
    df[col] = pd.to_numeric(df[col])
    df[col].fillna(df[col].mean(), inplace=True)
    
#%% encode ordinal data
cleanup_nums = {"num-of-doors":     {"four": 4, "two": 2},
                "num-of-cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}
df = df.replace(cleanup_nums)

from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
df["make_code"] = ord_enc.fit_transform(df[["make"]])
df=df.drop(columns=['make'])
#%% encode cathegorical data
df = pd.get_dummies(df, columns=["body-style", "drive-wheels"], prefix=["body", "drive"])

df["OHC_Code"] = np.where(df["engine-type"].str.contains("ohc"), 1, 0)

#%% create test train split

# only use selected attrs
attrs = ['symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
           'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price', 'num-of-doors', 'num-of-cylinders', 'body_convertible', 
           'body_hardtop', 'body_hatchback', 'body_sedan', 'body_wagon', 'drive_4wd', 'drive_fwd', 'drive_rwd', 'OHC_Code', 'make_code']

df = df[attrs]

#%%
x = np.array(df.drop('price',axis = 1))
y = np.array(df['price'])

from sklearn.model_selection import train_test_split
x_train,x_vali,y_train,y_vali = train_test_split(x,y,test_size = 0.1,random_state=42)
x_train = np.array(x_train).astype("float")
x_vali = np.array(x_vali).astype("float")
y_train = np.array(y_train).astype("float")
y_vali = np.array(y_vali).astype("float")


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# feed[attrs]=scaler.fit_transform(feed[attrs])

def print_mymetrics(scores, metrics,target):
    for metric in metrics:
        print("\n",metric)
        print("min: ",scores['test_'+metric].min())
        print("mean: ",scores['test_'+metric].mean())
        print("max:", scores['test_'+metric].max())
    print("\ntarget_mean_val", target.mean())


from sklearn.model_selection import cross_validate
#metrics = ['r2', 'max_error', 'neg_mean_absolute_error']
metrics = ['r2']
#%%
reg0 = M5regressor(smoothing=True, 
                  n_attr_leaf=4, 
                  max_depth=10, 
                  k=1.0,
                  pruning=True,
                  optimize_models=False,
                  incremental_fit=False,
                  prune_set=[x_vali, y_vali[:,None]])
scores0 = cross_validate(reg0, x_train, y_train[:,None], cv=10, scoring=metrics)
#%%
print_mymetrics(scores0, metrics, y_train)

#%%
reg1 = Const_regressor(n_attr_leaf=4, 
                       max_depth=15,
                       smoothing=False,
                       pruning=True,
                       k=0.5,
                       prune_set=[x_vali, y_vali[:,None]])
scores1 = cross_validate(reg1, x_train, y_train[:,None], cv=10, scoring=metrics)
#%%
print_mymetrics(scores1, metrics, y_train)
#%%
regr = LinearTreeRegressor(base_estimator=LinearRegression())
regr.fit(x_train, y_train)  # supports also multi-target and sample_weights
print(regr.score(x_train, y_train))

