#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the dataset into pandas

df=pd.read_csv("data.csv")


# In[3]:


#first few rows of the dataset

df.head()


# In[4]:


df['Age'].plot.hist()


# In[5]:


np.power(df['Age'],1/3).plot.hist()


# In[ ]:


bins=[0,15,80]

group=['children', 'Adult']


# In[ ]:


df['type']=pd.cut(df['Age'],bins,label=group)


# In[9]:


df.head()


# In[ ]:





# In[ ]:





# In[ ]:




