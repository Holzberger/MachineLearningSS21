#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 02:51:55 2021

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

df = pd.read_csv('../../../checkdatasets/housing.csv')
df=df.drop(["id","url","region_url","image_url","description"],axis=1)
df=df.drop(["state"],axis=1)

df['laundry_options'] = df['laundry_options'].fillna(df['laundry_options'].mode()[0])
df['parking_options'] = df['parking_options'].fillna(df['parking_options'].mode()[0])
df['lat'] = df['lat'].fillna(df['lat'].mean())
df['long'] = df['long'].fillna(df['long'].mean())



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["region"]=le.fit_transform(df["region"])
df["type"]=le.fit_transform(df["type"])
df["laundry_options"]=le.fit_transform(df["laundry_options"])
df["parking_options"]=le.fit_transform(df["parking_options"])




x=df.drop(columns=["price"])
y=df["price"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state=42)
x_train = np.array(x_train).astype("float")
x_test = np.array(x_test).astype("float")
y_train = np.array(y_train).astype("float")
y_test = np.array(y_test).astype("float")

from sklearn.metrics import mean_absolute_error
reg = Const_regressor(n_attr_leaf=4, max_depth=15)
reg.fit(x_train,y_train[:,None])
predictions = reg.predict(x_test)
print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
print("mean target :", np.mean(y))

from sklearn.linear_model import LinearRegression
dummy = LinearRegression().fit(x_train,y_train)
predictions = dummy.predict(x_test)
print("mean_absolute_error is : " , mean_absolute_error(y_test, predictions))
print("mean target :", np.mean(y))
