#!/usr/bin/env python
# coding: utf-8

# Fill valid code/values in place of blanks. 

# In[23]:


# import required libraries
import pandas as pd
import numpy as np


# In[ ]:


scores = [29,27,14,23,29,10]

# find the mean of all items of the list 'scores'
np.mean(____)


# In[ ]:


# find the median of all items of the list 'scores'
np.____(scores)


# In[ ]:


from statistics import mode

fruits = ['apple', 'grapes', 'orange', 'apple']

# find mode of the list 'fruits'
mode(____)


# In[ ]:


from random import sample
data = sample(range(1,100),50) # generating a list 50 random integers

# find variance of data
np.____(data)


# In[ ]:


# find standard deviation 
np.____(data)


# ### Please download the file "data_statistics.csv".

# In[73]:


# read data_python.csv using pandas
mydata = pd.read_csv("data_statistics.csv")


# In[ ]:


# print first few rows of mydata
mydata.head()


# In[ ]:


# plot histogram for 'Item_Outlet_Sales'
plt.____(mydata['Item_Outlet_Sales'])
plt.show()


# In[ ]:


# increadse no. of bins to 20
plt.____(mydata['Item_Outlet_Sales'], bins=____)
plt.show()


# In[ ]:


# find mean and median of 'Item_Weight'
np.____(mydata['Item_MRP']), np.____(mydata['Item_MRP'])


# In[ ]:


# find mode of 'Outlet_Size'
mydata['Outlet_Size'].____


# In[ ]:


# frequency table of 'Outlet_Type'
mydata['Outlet_Type'].____


# In[81]:


# mean of 'Item_Outlet_Sales' for 'Supermarket Type2' outlet type
np.mean(mydata['Item_Outlet_Sales'][mydata['Outlet_Type'] == 'Supermarket Type2'])


# In[82]:


# mean of 'Item_Outlet_Sales' for 'Supermarket Type3' outlet type
np.mean(mydata['Item_Outlet_Sales'][mydata['Outlet_Type'] == 'Supermarket Type3'])


# In[ ]:


# 2 sample independent t-test 
from scipy import stats
stats.____(mydata['Item_Outlet_Sales'][mydata['Outlet_Type'] == 'Supermarket Type2'], mydata['Item_Outlet_Sales'][mydata['Outlet_Type'] == 'Supermarket Type3'])

