#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy

file=pd.read_csv('C:/Users/gnbae/Downloads/heart_failure_dataset.csv')

X=file.drop(columns=['DEATH_EVENT'])

y=file['DEATH_EVENT']

model=DecisionTreeClassifier()
model.fit(X,y)



prediction=model.predict([[74,1,580,1,20,1,265000,2,130,1,1,5],[50,0,196,0,45,0,395000,1.6,136,1,1,285]])
prediction


# In[18]:





# In[ ]:




