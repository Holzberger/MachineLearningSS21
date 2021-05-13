#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:48:50 2021

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



# function that imputes a dataframe 
def impute_knn(df):
    
    ''' inputs: pandas df containing feature matrix '''
    ''' outputs: dataframe with NaN imputed '''
    # imputation with KNN unsupervised method

    # separate dataframe into numerical/categorical
    ldf = df.select_dtypes(include=[np.number])           # select numerical columns in df
    ldf_putaside = df.select_dtypes(exclude=[np.number])  # select categorical columns in df
    # define columns w/ and w/o missing data
    cols_nan = ldf.columns[ldf.isna().any()].tolist()         # columns w/ nan 
    cols_no_nan = ldf.columns.difference(cols_nan).values     # columns w/o nan 

    for col in cols_nan:                
        imp_test = ldf[ldf[col].isna()]   # indicies which have missing data will become our test set
        imp_train = ldf.dropna()          # all indicies which which have no missing data 
        model = KNeighborsRegressor(n_neighbors=5)  # KNR Unsupervised Approach
        knr = model.fit(imp_train[cols_no_nan], imp_train[col])
        ldf.loc[df[col].isna(), col] = knr.predict(imp_test[cols_no_nan])
    
    return pd.concat([ldf,ldf_putaside],axis=1)


df = pd.read_csv('../datasets/housing.csv')

df2 = impute_knn(df)

del df2['total_bedrooms']
del df2['total_rooms']
del df2['ocean_proximity']

df3 = df2#df2[['population','housing_median_age', 'median_income','median_house_value']]

#trdata,tedata = train_test_split(df3,test_size=0.3,random_state=43)




X = np.array(df2)
#XT = np.array(tedata)

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X=scaler.fit_transform(X)
#XT=scaler.fit_transform(XT)

# reg = M5regressor(smoothing=True, n_attr_leaf=5, max_depth=0, k=15.0)
# reg.fit(X[:,:-1],X[:,-1][:,None])
# reg.prune(X[:,:-1], X[:,-1][:,None], optimize_models=True)
# predictions = reg.predict(np.array(XT[:,:-1]))
# print("r2_score is : " , r2_score(XT[:,-1], predictions))


#print(score(XT[:,-1], predictions))

reg = Const_regressor(n_attr_leaf=10, max_depth=10)
reg.fit(X[:,:-1],X[:,-1][:,None])
predictions = reg.predict(np.array(X[:,:-1]))
print("r2_score is : " , r2_score(X[:,-1], predictions))

# from sklearn.model_selection import cross_val_score
# reg = M5regressor(smoothing=True, n_attr_leaf=140, max_depth=5, 
#                   k=400.0,pruning=True,optimize_models=True,incremental_fit=False)
# scores = cross_val_score(reg, X[:,:-1],X[:,-1][:,None], cv=10, scoring='r2')
# print(scores.min())
# print(scores.mean())
# print(scores.max())










# m=25
# x_pos = np.linspace(0, 1, m)
# y_pos = np.linspace(0, 1, m)
# x_pos2d, y_pos2d = np.meshgrid(x_pos, y_pos)
# X_pos = np.hstack((x_pos2d.reshape(m*m,1), y_pos2d.reshape(m*m,1)))
# Z = reg.predict(X_pos).reshape(m,m)

# from matplotlib import cm
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(x_pos2d, y_pos2d, Z, cmap=cm.coolwarm, linewidth=0,alpha=1)
# npoints=5000
# surf = ax.scatter(X[:npoints,0], X[:npoints,1], X[:npoints,-1], cmap=cm.coolwarm,marker=".",alpha=0.1)



#sns.regplot(x = XT[:,-1], y = predictions)