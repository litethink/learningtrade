#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


# In[2]:


def load_data(folder):
    dic={}
    for filename in os.listdir(folder):
        data=pd.read_csv(os.path.join(folder,filename))
        dic[data['Symbol'][0]]=data['Close Price'][0:493].values
    return dic


# In[5]:


# Creating dataframe with 30 different stocks,10 from each cap

data=pd.DataFrame(load_data('data'))
data.head()


# In[6]:


# Calculating annual average return and volatility of all 30 stocks

daily_return=data.pct_change()*100
daily_return=daily_return.dropna()

ann_mean_rtn=daily_return.mean()*252

vol=daily_return.std()*np.sqrt(252)


# In[7]:


# plot of volatility vs annual average return

plt.scatter(vol,ann_mean_rtn)


# In[8]:


# preparation of input to be given in clustering algorithmn

names=pd.Series(vol.index).set_axis(vol.index)
X=pd.concat([names,vol,ann_mean_rtn],axis='columns',keys=['Symbol','Volatility','return'])


# In[9]:


X


# In[12]:


# finding optimum number of cluster using elbow method

from sklearn.cluster import KMeans

w=[]
for i in range(1,20):
    km=KMeans(n_clusters=i)
    km.fit(X.iloc[:,[1,2]])
    w.append(km.inertia_)
    
plt.plot(range(1,20),w,marker='o')  


# In[13]:


# found optimum number of cluster is 6 approximatly

km=KMeans(n_clusters=6)
y_pred=km.fit_predict(X.iloc[:,[1,2]])
X['Cluster']=y_pred
X=X.set_index(np.arange(len(vol)))


# In[14]:


# Dataframe showing eack stock and associated cluster

print(X)


# In[15]:


# plotting of Clusters

plt.figure(figsize=(15,8))
plt.scatter(X.iloc[y_pred==0,1],X.iloc[y_pred==0,2],c='r',label='Cluster 0',s=100)
plt.scatter(X.iloc[y_pred==1,1],X.iloc[y_pred==1,2],c='g',label='Cluster 1',s=100)
plt.scatter(X.iloc[y_pred==2,1],X.iloc[y_pred==2,2],c='b',label='Cluster 2',s=100)
plt.scatter(X.iloc[y_pred==3,1],X.iloc[y_pred==3,2],c='c',label='Cluster 3',s=100)
plt.scatter(X.iloc[y_pred==4,1],X.iloc[y_pred==4,2],c='m',label='Cluster 4',s=100)
plt.scatter(X.iloc[y_pred==5,1],X.iloc[y_pred==5,2],c='y',label='Cluster 5',s=100)
plt.legend()
plt.xlabel('Volatility')
plt.ylabel('Return')
for name,vol,rtn in zip(X['Symbol'],X['Volatility'],X['return']):
    plt.text(vol,rtn,name)


# In[ ]:




