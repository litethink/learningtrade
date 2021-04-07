#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#calculation of annual return and annual standard deviation of chosen stock


# In[6]:


df=pd.read_csv('JUBLFOOD.csv')
df.head()


# In[7]:


df['Series'].value_counts()


# In[8]:


df=df[df['Series']!='BL']


# In[9]:

#(values[i] - values[i-1]) /values[i-2]
returns=df['Close Price'].pct_change()*100

returns.dropna()

annual_mean=returns.mean()*252

annual_std=returns.std()*np.sqrt(252)


# In[11]:


# preparing portfolio and calculating annual return and volatility of entire portfolio

tata=pd.read_csv('TATAPOWER.csv')
axis=pd.read_csv('AXISBANK.csv')
ashoka=pd.read_csv('ASHOKA.csv')
fortis=pd.read_csv('FORTIS.csv')
tcs=pd.read_csv('TCS.csv')

ashoka=ashoka.iloc[0:494]
axis=axis.iloc[0:494]
tata=tata.iloc[0:494]
fortis=fortis.iloc[0:494]
tcs=tcs.iloc[0:494]

data=pd.DataFrame({'tata':tata['Close Price'],
                   'axis':axis['Close Price'],
                   'ashoka':ashoka['Close Price'],
                   'fortis':fortis['Close Price'],
                   'tcs':tcs['Close Price']})


returns=data.pct_change()*100
returns=returns.dropna()


# In[12]:


w=np.array([[0.2,0.2,0.2,0.2,0.2]]).reshape(-1,1)

ann_mean_return=returns.mean()*252

port_return=w.T@ann_mean_return

port_var=w.T@(returns.cov()@w)


# In[13]:


# preparing a scatter plot for differing weights of individaul stocks in portfolio
# mark two portfolios, one with max sharpe ratio and other with lowest volatility 

weights=np.random.random((25000,5))
s=np.sum(weights,axis=1)

r=[]
v=[]
sharpe=[]
for i in range(len(weights)):
    weights[i,:]=weights[i,:]/s[i]
    r.append(weights[i,:].T@ann_mean_return)
    v.append(weights[i,:].T@(returns.cov()@weights[i,:]))
    sharpe.append(r[i]/v[i])

port_data=pd.DataFrame({'w1':weights[:,0],
                        'w2':weights[:,1],
                        'w3':weights[:,2],
                        'w4':weights[:,3],
                        'w5':weights[:,4],
                        'return':r,
                        'risk':v,
                        'sharpe':sharpe})


# In[14]:


# portfolio with maximum sharpe ratio

max_sharpe=port_data[port_data['sharpe']==port_data['sharpe'].max()]


# In[15]:


# portfolio with lowest volatility

min_vol=port_data[port_data['risk']==port_data['risk'].min()]


# In[18]:


plt.figure(figsize=(15,8))
plt.scatter(port_data['risk'],port_data['return'],c=sharpe)
plt.colorbar()
#plt.scatter(max_sharpe['risk'],max_sharpe['return'],c='r',marker='*',s=500)
plt.scatter(min_vol['risk'],min_vol['return'],c='b',marker='*',s=500)
plt.xlabel('Risk')
plt.ylabel('Return')


# In[ ]:




