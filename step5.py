# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:16:17 2020

@author: selen
"""
"""
K Nearest Neighborhood
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("veriler.csv")
part_x= data.iloc[:,1:4]
part_y=data.iloc[:,-1:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(part_x,part_y, test_size=0.33)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5, metric="minkowski")
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)
print(y_predict)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
print(cm)
"""
Naive Bayes
"""
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_predict=gnb.predict(x_test)
cm=confusion_matrix(y_test,y_predict)
print(cm)
from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB()
bnb.fit(x_train,y_train)
y_predict=bnb.predict(x_test)
cm=confusion_matrix(y_test,y_predict)
print(cm)
"""
Decision Tree
"""
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(x_train,y_train)
y_predict=dtc.predict(x_test)
cm=confusion_matrix(y_test,y_predict)
print(cm)
"""
Random Forrest
"""
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(x_train,y_train)
y_predict=rfc.predict(x_test)
cm=confusion_matrix(y_test,y_predict)
print(cm)





