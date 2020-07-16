# -*- coding: utf-8 -*-
"""

@author: selen
"""


"""
k-means
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

x = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(x)
#kyı makine bulabilir
print(kmeans.inertia_)