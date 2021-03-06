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
from sklearn.impute import KNNImputer
from sklearn.utils import shuffle

from treend import *
from reg_tree import *

df = pd.read_csv('../datasets/Automobile_data.csv')

# Converting price column type to numeric & Dropping '?' entries from price column 
"""
df['price'] = df['price'].replace('?',np.nan)
df['price'] = pd.to_numeric(df['price'])
df = df[df['price'].notna()]

# Dropping '?' from num-of-doors column 
df['num-of-doors'] = df['num-of-doors'].replace('?',np.nan)
df = df[df['num-of-doors'].notna()]

num_col = ['normalized-losses', 'bore',  'stroke', 'horsepower', 'peak-rpm']
for col in num_col:
    df[col] = df[col].replace('?', np.nan)
    df[col] = pd.to_numeric(df[col])
    df[col].fillna(df[col].mean(), inplace=True)
    
    
cleanup_nums = {"num-of-doors":     {"four": 4, "two": 2},
                "num-of-cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}
df = df.replace(cleanup_nums)

# what about
df = pd.get_dummies(df, columns=["body-style", "drive-wheels"], prefix=["body", "drive"])
df.head()

df["OHC_Code"] = np.where(df["engine-type"].str.contains("ohc"), 1, 0)
df[["make", "engine-type", "OHC_Code"]].head()

from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
df["make_code"] = ord_enc.fit_transform(df[["make"]])
df[["make", "make_code"]].head(11)
=

attrs = ['symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
           'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price', 'num-of-doors', 'num-of-cylinders', 'body_convertible', 
           'body_hardtop', 'body_hatchback', 'body_sedan', 'body_wagon', 'drive_4wd', 'drive_fwd', 'drive_rwd', 'OHC_Code', 'make_code']

feed = df[attrs]
"""

df=df.replace('?',np.nan)
cleanup_nums = {"num-of-doors":     {"four": 4, "two": 2},
                "num-of-cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}
df = df.replace(cleanup_nums)
df = pd.get_dummies(df, columns=["make", "fuel-type", "num-of-doors","fuel-system", "aspiration",  "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders"  ], prefix=["make", "fuel_type", "doors", "fuel_sys", "aspi",  "body", "drive", "eng_loc", "eng_type", "cylinders"], drop_first= True)
#df = pd.get_dummies(df, columns=["make", "fuel-type", "num-of-doors", "aspiration",  "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders"  ], prefix=["make", "fuel_type", "doors",  "aspi",  "body", "drive", "eng_loc", "eng_type", "cylinders" ])

df_imputed=df
imputer = KNNImputer(n_neighbors=4)
df_imputed= pd.DataFrame(imputer.fit_transform(df_imputed), index=df.index, columns=df.columns)


attrs = list(df_imputed.columns.values)

feed = df_imputed[attrs]

#feed = df



"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
feed[attrs]=scaler.fit_transform(feed[attrs])
"""
"""

#df_train_x = feed.drop('price',axis = 1)
#df_train_x.describe()
#df_train_y = feed['price']
#df_train_y.describe

"""
X = np.array(feed.drop('price',axis = 1))
y = np.array(feed['price'])

# reg = M5regressor(smoothing=True, n_attr_leaf=4, max_depth=15, k=15.0)
# reg.fit(np.array(x_train), np.array(y_train)[:,None])
# reg.prune(np.array(x_test), np.array(y_test)[:,None], optimize_models=False)
# predictions = reg.predict(np.array(x_test))
# print("r2_score is : " , r2_score(y_test, predictions))

#sns.regplot(x = y_test, y = predictions)

X,  y = shuffle(X, y, random_state=42)

from sklearn.model_selection import cross_val_score
reg = M5regressor(smoothing=True, n_attr_leaf=4, max_depth=10,
                  k=20.0,pruning=True,optimize_models=True,incremental_fit=True)
scores = cross_val_score(reg, X,y[:,None], cv=5, scoring='r2')
print(scores.min())
print(scores.mean())
print(scores.max())
print(scores)

reg2 = Const_regressor(n_attr_leaf=4, max_depth=10)
scores2 = cross_val_score(reg2, X,y[:,None], cv=5, scoring='r2')
print(scores2.min())
print(scores2.mean())
print(scores2.max())
print(scores2)


