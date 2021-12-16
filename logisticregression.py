#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing required libraries

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlin', 'inline')
import numpy as np


# In[ ]:


data=pd.read_csv("titanic.csv")


# In[ ]:


data.head()


# In[ ]:


data['Survived'].value_counts()


# In[ ]:


data=pd.get_dummies(data)


# In[ ]:


data.fillna(0,inplace=True)


# In[ ]:


data.shape()


# In[ ]:


train=data[0:699]


# In[ ]:


test=data[700:890]


# In[ ]:


x_train=train.drop('Survived', axis=1)


# In[ ]:


true_p = test['Survived']


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg=LogisticRegression()


# In[ ]:


logreg.fit(x_train,y_train)


# In[ ]:


pred=logreg.predict(x_test)


# In[ ]:


pred


# In[ ]:


logreg.score(x_test,true_p)


# In[ ]:


logreg.score(x_train,y_train)

