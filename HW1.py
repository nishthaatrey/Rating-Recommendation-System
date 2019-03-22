#!/usr/bin/env python
# coding: utf-8


from pyspark.ml.recommendation import ALS
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark import sql
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id
import math



spark = SparkSession.builder.appName("hw1").getOrCreate()

all_lines = spark.read.text("train.dat").rdd
divs = all_lines.map(lambda row: row.value.split("\t"))
row_rdd = divs.map(lambda a: Row(userId=int(a[0]),movieId=int(a[1]), rating=float(a[2]), timestamp=int(a[3])))
df = spark.createDataFrame(row_rdd)
df.show(10)

als = ALS(maxIter=8,regParam=0.085, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="nan")
model= als.fit(df)

all_lines_test = spark.read.text("test.dat").rdd
divs_test = all_lines_test.map(lambda row: row.value.split("\t"))
row_rdd_test = divs_test.map(lambda a: Row(userId=int(a[0]),movieId=int(a[1])))
df_test = spark.createDataFrame(row_rdd_test)
df_test.show(10)

res = df_test.withColumn("col_id", monotonically_increasing_id())
res.show(10)
predictions = model.transform(res)
predictions.show(20)
sample= predictions.sort(predictions.col_id)
sample1= sample.select(sample.col_id,sample.movieId,sample.userId,sample.prediction)
sample1.show(10)
lis = sample1.select("prediction").rdd.flatMap(list).collect()

final=[]

for rating in lis:
    if (math.isnan(rating)==True):
        final.append(2)
    else:
        final.append(int(round(rating, 0)))

f = open("output1.txt","w")
for r in final:
    f.write(str(r)+"\n")



