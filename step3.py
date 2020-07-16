# -*- coding: utf-8 -*-
"""

@author: selen
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
"""
Multiple Linear Regression
"""
# ülke,cinsiyet,kilo, yaş değişkenlerinden boy tahmin etme
data=pd.read_csv("veriler.csv")
print(data)
cinsiyet=data.iloc[:,-1:].values
ülke=data.iloc[:,0:1].values
lb=LabelEncoder()
ohe=OneHotEncoder()
ülke=ohe.fit_transform(ülke).toarray()
cinsiyet=ohe.fit_transform(cinsiyet).toarray() #labelEncoder daha iyi olur
dataFrame_1=pd.DataFrame(data=ülke,index=range(22),columns=["tr","fr","us"])
part_x=pd.concat([dataFrame_1,data[["kilo","yas"]],pd.DataFrame(data=cinsiyet[:,0:1],index=range(22),columns=["cinsiyet"])],axis=1)
part_y=data[["boy"]]
x_train,x_test,y_train,y_test=train_test_split(part_x,part_y,test_size=0.33)
"""
*****************
"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_tahmin=regressor.predict(x_test)

"""
p-value // backward elimination
"""
"""
Başarı Ölçütleri Giriş
"""

import statsmodels.api as sm

X=np.append(arr= np.ones((22,1)).astype(int), values=part_x,axis=1)
"""
her sütunun ne kadar etkilediği bulunur,sistemi en çok bozan çıkarılır 
"""
X_list=part_x.iloc[:,[0,1,2,3,4,5]].values
r=sm.OLS(endog=part_y,exog=X_list).fit()
print(r.summary())
"""
4. column çıkarıldı  // elendi
"""

X_list=part_x.iloc[:,[0,1,2,3,5]].values
r=sm.OLS(endog=part_y,exog=X_list).fit()
print(r.summary())

#%%
"""
polynomial regression
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("maaslar.csv")
# veri eğitim ve test kümesi olarak ayrılmadan kullanılacak
eğitim_seviyesi=data.iloc[:,1:2]
maas=data.iloc[:,-1:]
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_regression=PolynomialFeatures(degree=2)
x_poly=poly_regression.fit_transform(eğitim_seviyesi.values)
lin_regression=LinearRegression()
lin_regression.fit(x_poly,maas)
plt.scatter(eğitim_seviyesi.values,maas.values,color="red")
plt.plot(eğitim_seviyesi.values,lin_regression.predict(x_poly),color="blue")
#a=lin_regression.predict(poly_regression.fit_transform(np.array([11]))) ÇALIŞMIYOR
#print(a) ÇALIŞMIYOR
"""
Decision Tree
"""
plt.show()
from sklearn.tree import DecisionTreeClassifier
decisionT_regressor=DecisionTreeClassifier(random_state=0)
decisionT_regressor.fit(eğitim_seviyesi.values,maas.values)
plt.scatter(eğitim_seviyesi.values,maas.values)
plt.plot(eğitim_seviyesi.values,decisionT_regressor.predict(eğitim_seviyesi.values),color="red")
"""
Random Forest
"""
plt.show()
from sklearn.ensemble import RandomForestRegressor
randomF_regressor=RandomForestRegressor(n_estimators=10,random_state=0)
randomF_regressor.fit(eğitim_seviyesi.values,maas.values)
plt.scatter(eğitim_seviyesi.values,maas.values)
plt.plot(eğitim_seviyesi.values,randomF_regressor.predict(eğitim_seviyesi.values),color="red")
"""
R-SQUARE METHOD: Evaluation Of Predictions
"""
from sklearn.metrics import r2_score
print("random forest r2 value: ")
print(r2_score(maas.values,randomF_regressor.predict(eğitim_seviyesi.values)))





















