# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:55:38 2020

@author: selen
"""
#%%
"""
LINEAR REGRESSION
"""
import pandas as pd
import numpy as np
veriler=pd.read_csv("satislar.csv")
#print(veriler)
aylar=veriler[["Aylar"]]
satislar=veriler[["Satislar"]]
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test= train_test_split(aylar,satislar,test_size=0.33,random_state=0 )
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train=sc.fit_transform(x_train)
Y_train=sc.fit_transform(y_train)   
X_test=sc.fit_transform(x_test)
Y_test=sc.fit_transform(y_test)
"""
************************
"""
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr=lr.fit(x_train,y_train)  #Scale edilmemiş verilerde tahmin
tahmin=lr.predict(x_test)
import matplotlib.pyplot as plt
x_train=x_train.sort_index()
y_train=y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)
