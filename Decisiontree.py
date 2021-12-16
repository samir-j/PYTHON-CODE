#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries 

import pandas as pd 
import numpy as np


# In[ ]:


data=pd.read_csv('data_cleaned.csv')


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


#seperating independent and dependent variables

x = data.drop(['Survived'], axis=1)
y = data['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 101, stratify=y)


# In[ ]:


train_y.value_counts()/len(train_y)


# In[ ]:


test_y.value_counts()/len(test_y)


# In[ ]:


#importing decision tree classifier 

from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf = DecisionTreeClassifier()


# In[ ]:


clf.fit(train_x,train_y)


# In[ ]:


clf.score(train_x, train_y)


# In[ ]:


clf.score(test_x, test_y)


# In[ ]:


clf.predict(test_x)

