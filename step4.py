# -*- coding: utf-8 -*-
"""


@author: selen
"""
"""
Logistic Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data=pd.read_csv("veriler.csv")
part_x= data.iloc[:,1:4]
part_y=data.iloc[:,-1:]
x_train,x_test,y_train,y_test=train_test_split(part_x,part_y, test_size=0.33)

from sklearn.linear_model import LogisticRegression

logistic_regression=LogisticRegression(random_state=0)
logistic_regression.fit(x_train,y_train)
y_predict=logistic_regression.predict(x_test)
print(y_predict)
print(y_test.values)
"""
Confusion Matrix: Karmaşıklık Matrisi
"""
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_predict,y_test)
print(cm)
