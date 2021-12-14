#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib.pyplot import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans


# In[ ]:


data=pd.read_csv("student_evaluation.csv")


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


pd.isnull(data).sum()


# In[ ]:


data.describe()


# In[ ]:


kmeans = KMeans(n_cluster=2)


# In[ ]:


kmeans.fit(data)


# In[ ]:


pred=kmeans.predict(data)


# In[ ]:


pred


# In[ ]:


pd.series(pred).value_counts()


# In[ ]:


kmeans.inertia_


# In[ ]:


kmeans.score(data)


# In[ ]:


SSE = []


# In[ ]:


for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster)
    kmean.fit(data)
    SSE.append(kmeans.inertia_)


# In[ ]:


frame = pd.Dataframe({'Cluster':range(1,20), 'SSE':SSE})


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data_scaled = scaler.fit_transform(data)


# In[ ]:


pd.Dataframe(data_scaled).describe()


# In[ ]:


SSE_scaled = []


# In[ ]:


for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster)
    kmean.fit(data_scaled)
    SSE.append(kmeans.inertia_)


# In[ ]:


frame_scaled = pd.Dataframe({'Cluster':range(1,20), 'SSE':SSE_scaled})
plt.plot(frame_scaled['Cluster'], frame_scaled['SSE'], marker='o')
plt.xlabel("Clusters")
plt.ylabel("SSE")


# In[ ]:


kmeans = KMeans(n_jobs = -1, n_clusters = 4)
kmean.fit(data_scaled)
pred = kmeans.predict(data_scaled)


# In[ ]:


pred


# In[ ]:


frame = pd.DataFrame(data_scaled)


# In[ ]:


frame['cluster'] = pred


# In[2]:


frame.loc[frame['cluster']==2,:]


# In[ ]:




