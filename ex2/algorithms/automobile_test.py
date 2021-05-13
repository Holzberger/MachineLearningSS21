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

from treend import *
from reg_tree import *
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

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# feed[attrs]=scaler.fit_transform(feed[attrs])

X = np.array(feed.drop('price',axis = 1))
y = np.array(feed['price'])




from sklearn.model_selection import cross_val_score
reg = M5regressor(smoothing=False, n_attr_leaf=4, max_depth=16, 
                  k=1.0,pruning=False,optimize_models=False,incremental_fit=False)
scores = cross_val_score(reg, X,y[:,None], cv=10, scoring='r2')
print(scores.min())
print(scores.mean())
print(scores.max())
print(scores)

