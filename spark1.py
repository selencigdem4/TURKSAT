# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:39:33 2020

@author: selen
"""
from pyspark.sql import SparkSession
spark= SparkSession.builder.appName("başlangıç").getOrCreate()
data=spark.read.format("csv").option("header", True).load("veriler.csv")
data.show(5)
data.select("ulke").show()
data.filter(data["yas"]<10).orderBy("yas",ascending=False).show()
data.groupBy("ulke").count().orderBy("count").show()
df=spark.createDataFrame(data=[(1,2,3)],schema=["a","b","c"])
df.show()



#%%
import findspark
"""
ÇALIŞMADI #kod örneği
"""
findspark.init("C:\spark")

from pyspark.sql import SparkSession

spark= SparkSession.builder.appName("first_try").getOrCreate()
data=spark.read.format("csv").option("header","true").load("ratings.csv")
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
data.show()
data.describe().show()
als=ALS(maxIter=5,regParam=0.01,userCol="userId",itemCol="movieId",ratingCol="rating")
training,test=data.randomSplit([0.7,0.3])
model=als.fit(training)
prediction=model.transform(test)
prediction.show()
#%%
from pyspark.sql import SparkSession

spark= SparkSession.builder.appName("veriler_pyspark").getOrCreate()
from pyspark.ml.regression import LinearRegression
data= spark.read.csv("veriler.csv", inferSchema=True,header=True)
data.show()
data.describe().show()
data.printSchema() #*******************************
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler=VectorAssembler(inputCols=["yas","kilo"], outputCol="features")
output=assembler.transform(data)
output.show()
output.select("features").show()
finalized_output=output.select("features","boy")

train,test=finalized_output.randomSplit([0.7,0.3])
lin_reg=LinearRegression(featuresCol="features",labelCol="boy")
lin_reg=lin_reg.fit(train)
tahmin=lin_reg.evaluate(test)
tahmin.predictions.show()










