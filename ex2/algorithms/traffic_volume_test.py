#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 19:36:28 2021

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

#%% encode remaining cathegorical attrs as one hot
df = pd.get_dummies(df, columns=["holiday", "weather_main", "weather_description"], prefix=["holiday", "weather_main", "weather_description"])

#%% create test train split
x=df.drop(columns=["traffic_volume"])
y=df["traffic_volume"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
x_train = np.array(x_train).astype("float")
x_test = np.array(x_test).astype("float")
y_train = np.array(y_train).astype("float")
y_test = np.array(y_test).astype("float")
x_train,x_vali,y_train,y_vali = train_test_split(x_train,y_train,test_size = 0.2,random_state=42)
#%%
# from sklearn.metrics import mean_absolute_error
# #reg0 = Const_regressor(n_attr_leaf=4, max_depth=15,smoothing=False)
# reg0 = Const_regressor(n_attr_leaf=4, max_depth=15,smoothing=False)
# reg0.fit(x_train,y_train[:,None])
# #%%
# reg0.prune(x_vali,y_vali[:,None])
# #%%
# reg0.smoothing=True
# reg0.k=0.5
# predictions = reg0.predict(x_test)
# print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
# print("mean target :", np.mean(y))
#%%

# from sklearn.model_selection import cross_val_score
# reg1 = M5regressor(n_attr_leaf=4, max_depth=15)
# reg1.incremental_fit=False
# reg1.fit(x_train,y_train[:,None])
# #%%
# optimize_models=True
# reg1.prune(x_vali,y_vali[:,None])
#%%
# reg1.smoothing=True
# reg1.k=200.0
# predictions = reg1.predict(x_test)
# print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
# print("mean target :", np.mean(y))
# #%%
# from sklearn.linear_model import LinearRegression
# dummy = LinearRegression().fit(x_train,y_train)
# predictions = dummy.predict(x_test)
# print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
# print("mean target :", np.mean(y))

# #%%
# from sklearn.tree import DecisionTreeRegressor
# reg2 = DecisionTreeRegressor(random_state=42,criterion="friedman_mse", max_depth=7)
# reg2.fit(x_train,y_train)
# #%%
# predictions = reg2.predict(x_test)
# print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
# print("mean target :", np.mean(y))



reg3 = LinearTreeRegressor(base_estimator=LinearRegression())
reg3.fit(x_train,y_train)  # supports also multi-target and sample_weights


