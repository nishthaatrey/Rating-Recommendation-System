#!/usr/bin/env python
# coding: utf-8

# In[58]:



from pyspark.ml.recommendation import ALS
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark import sql
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id
import math


# In[59]:


spark = SparkSession.builder.appName("hw1").getOrCreate()


# In[60]:


all_lines = spark.read.text("train.dat").rdd


# In[61]:


divs = all_lines.map(lambda row: row.value.split("\t"))


# In[ ]:





# In[62]:


row_rdd = divs.map(lambda a: Row(userId=int(a[0]),movieId=int(a[1]), rating=float(a[2]), timestamp=int(a[3])))


# In[ ]:





# In[63]:


df = spark.createDataFrame(row_rdd)
df.show(10)


# In[64]:


als = ALS(maxIter=13,regParam=0.090, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="nan")


# In[65]:


model= als.fit(df)


# In[66]:


all_lines_test = spark.read.text("test.dat").rdd


# In[67]:


divs_test = all_lines_test.map(lambda row: row.value.split("\t"))


# In[ ]:





# In[68]:


row_rdd_test = divs_test.map(lambda a: Row(userId=int(a[0]),movieId=int(a[1])))


# In[69]:


df_test = spark.createDataFrame(row_rdd_test)
df_test.show(10)

res = df_test.withColumn("col_id", monotonically_increasing_id())
res.show(10)


# In[70]:


predictions = model.transform(res)


# In[71]:


predictions.show(20)


# In[72]:



sample= predictions.sort(predictions.col_id)
sample1= sample.select(sample.col_id,sample.movieId,sample.userId,sample.prediction)
sample1.show(10)


# In[73]:


lis = sample1.select("prediction").rdd.flatMap(list).collect()


# In[74]:


final=[]

for rating in lis:
    if (math.isnan(rating)==True):
        final.append(2)
    else:
        final.append(int(round(rating, 0)))


# In[75]:


final


# In[76]:


f = open("output3.txt","w")
for r in final:
    f.write(str(r)+"\n")


# In[ ]:




