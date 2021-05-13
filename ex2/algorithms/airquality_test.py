#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:12:48 2021

@author: fabian
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from treend import *
from reg_tree import *

from os import listdir
from os.path import isfile, join



frames= []
for file in files:
    df = pd.read_csv(data_path+file)
    frames.append(df)

df = pd.concat(frames)

#%%

# only predict o3
df=df.drop(columns=['No','PM2.5', 'PM10', 'SO2', 'NO2', 'CO'])

print("Missing vals:")
print(df.isnull().any())

# inset most common value for missings:
df['wd'] = df['wd'].fillna(df['wd'].mode()[0])

missing_list = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
for name in missing_list:
    df[name] = df[name].fillna(df[name].mean())

df = df[df['O3'].notna()]

print("Missing vals:")
print(df.isnull().any())

#%%

# encode columns
df = pd.get_dummies(df, columns=["wd", "station"], prefix=["wd", "stat"])

x=df.drop(columns=["O3"])
y=df["O3"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
x_train = np.array(x_train).astype("float")
x_test = np.array(x_test).astype("float")
y_train = np.array(y_train).astype("float")
y_test = np.array(y_test).astype("float")


#%%
from sklearn.metrics import mean_absolute_error
reg0 = Const_regressor(n_attr_leaf=4, max_depth=8,smoothing=False)
reg0.fit(x_train,y_train[:,None])
#%%
reg0.smoothing=False
reg0.k=0.0
predictions = reg0.predict(x_test)
print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
print("mean target :", np.mean(y))
#%%

from sklearn.model_selection import cross_val_score
reg1 = M5regressor(smoothing=False, n_attr_leaf=4, max_depth=8, 
                  k=1.0,pruning=False,optimize_models=False,incremental_fit=False)
reg1.fit(x_train,y_train[:,None])
#%%
reg1.smoothing=True
reg1.k=150.0
predictions = reg1.predict(x_test)
print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
print("mean target :", np.mean(y))


#%%
from sklearn.linear_model import LinearRegression
dummy = LinearRegression().fit(x_train,y_train)
predictions = dummy.predict(x_test)
print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
print("mean target :", np.mean(y))

#%%
from sklearn.tree import DecisionTreeRegressor
reg2 = DecisionTreeRegressor(random_state=42,criterion="friedman_mse", max_depth=20)
reg2.fit(x_train,y_train)
#%%
predictions = reg2.predict(x_test)
print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
print("mean target :", np.mean(y))



