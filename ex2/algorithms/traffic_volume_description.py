#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:06:05 2021

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

#%% replace zero kelvin samples by mean
error_ind = np.where(df['temp']<1.0)[0]
print("Temperature measurement errors: ", error_ind.shape[0] )
df['temp'][error_ind] = np.mean(df['temp'][df['temp']>1.0])
#%% make holiday binary
df['holiday'][df['holiday']!="None"] = 1
df['holiday'][df['holiday']=="None"] = 0
#%% target value over attributes
alpha = 0.007
fig, ax = plt.subplots(3,3, figsize = (13,9))

ax[0,0].set_title("hour of day",fontsize=8,fontweight='bold')
ax[0,0].set_ylabel("traffic volume",fontsize=8,fontweight='bold')
ax[0,0].plot(df['hour'], df['traffic_volume'], ".",alpha=alpha)

ax[0,1].set_title("day of week",fontsize=8,fontweight='bold')
ax[0,1].plot(df['day_week'], df['traffic_volume'], ".",alpha=alpha)

ax[0,2].set_title("day of year",fontsize=8,fontweight='bold')
ax[0,2].plot(df['day_year'], df['traffic_volume'], ".",alpha=alpha)


ax[1,0].set_title("temperature [Kelvin]",fontsize=8,fontweight='bold')
ax[1,0].set_ylabel("traffic volume",fontsize=8,fontweight='bold')
ax[1,0].plot(df['temp'], df['traffic_volume'], ".",alpha=alpha)

ax[1,1].set_title("rain 1h",fontsize=8,fontweight='bold')
ax[1,1].semilogx(df['rain_1h'], df['traffic_volume'], ".",alpha=1)

ax[1,2].set_title("snow 1h",fontsize=8,fontweight='bold')
ax[1,2].plot(df['snow_1h'], df['traffic_volume'], ".",alpha=1)

ax[2,0].set_title("holiday",fontsize=8,fontweight='bold')
ax[2,0].set_ylabel("traffic volume",fontsize=8,fontweight='bold')
ax[2,0].plot(df['holiday'], df['traffic_volume'], ".",alpha=1)

ax[2,1].set_title("weather main",fontsize=8,fontweight='bold')
ax[2,1].plot(df['weather_main'], df['traffic_volume'], ".",alpha=alpha)

ax[2,2].set_title("weather description",fontsize=8,fontweight='bold')
labels, levels = pd.factorize(df['weather_description'])
ax[2,2].plot(labels, df['traffic_volume'], ".",alpha=alpha)

plt.sca(ax[2, 1])
plt.xticks(rotation = 25, ha="right")
#%% target distribution
plt.figure(figsize=(5,5))
plt.hist(df['traffic_volume'], bins=np.linspace(df['traffic_volume'].min(),df['traffic_volume'].max(), 50), color='blue', edgecolor='k', alpha=0.65)
plt.xticks(fontsize=11)
plt.title("Distributon of the traffic volume attribute",fontsize=12,fontweight='bold')
plt.ylabel("number of instances",fontsize=12,fontweight='bold')
plt.xlabel("traffic volume",fontsize=12,fontweight='bold')
plt.grid(True, which="both", ls="-")
#plt.savefig("./traffic_vol.svg")
#%% encode remaining cathegorical attrs as one hot
df = pd.get_dummies(df, columns=["holiday", "weather_main", "weather_description"], prefix=["holiday", "weather_main", "weather_description"])
#%%







