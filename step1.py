# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:30:38 2020

@author: selen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data =pd.read_csv("veriler.csv")
print(data)
boy_kilo= data[["boy","kilo"]]
print(boy_kilo)
#%% 
"""
EKSİK VERİLERİ TAMAMLAMA
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

eksik_veriler= pd.read_csv("eksikveriler.csv")
print(eksik_veriler)
imputer= SimpleImputer(missing_values=np.nan, strategy = 'mean' ) #np.nan="NaN"
numerical_data= eksik_veriler.iloc[:,1:4]
imputer=imputer.fit(numerical_data)
eksik_veriler.iloc[:,1:4]=imputer.transform(numerical_data)
print(eksik_veriler.iloc[:,1:4].values)
#%%
"""
LabelEncoder ve OneHotEncoder
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data =pd.read_csv("veriler.csv")
print(ülke)
ülke=data.iloc[:,0:1].values #data.iloc[:,0:1].values
cinsiyet=data.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
#ülke polynomial -> OHE
#cinsiyet binomial -> LE
ohe=OneHotEncoder() 
cinsiyet=le.fit_transform(cinsiyet)
ülke=ohe.fit_transform(ülke).toarray()
print(cinsiyet)
print(ülke)
"""
dataframelerin birleştirilmesi:
"""
ülke=pd.DataFrame(data=ülke,index=range(22),columns=['tr','fr','us' ])
print(sonuç)
yaş=data[['yas']] #zaten dataframe olarak alınmış
print(yaş)
concat=pd.concat([ülke,yaş],axis=1) #satır bazlı birleştirme
print(concat)
#%%
"""
veriyi test ve training olarak ayırma
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
#1/3 for test 2/3 for training
data=pd.read_csv("veriler.csv")
cinsiyet=data[["cinsiyet"]]

ülke=data.iloc[:,0:1].values 
ohe=OneHotEncoder() 
ülke=ohe.fit_transform(ülke).toarray()
kalan_data=data.iloc[:,1:4].values
kalanDf=pd.DataFrame( data= kalan_data,index=range(22),columns=["boy","kilo","yas"] )
ülkeDf=pd.DataFrame(data=ülke , columns=["tr","fr","us"] , index=range(22) )

sonuç=pd.concat([ülkeDf,kalanDf],axis=1)

x_train,x_test,y_train,y_test= train_test_split(sonuç,cinsiyet,test_size=0.33,random_state=0 )
print(x_train)
"""
FEATURE SCALING(ÖZNİTELİK ÖLÇEKLEME)
"""
from sklearn.preprocessing import StandardScaler
#diğeri normalisation ama standradisation daha sık tercih edilir özellikle outlier(uç değer)larda

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print(x_test)










  
   






