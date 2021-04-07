#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


gold=pd.read_csv('GOLD.csv') 
gold.head()


# In[4]:


gold.dtypes


# In[5]:


gold.isnull().sum()


# In[6]:


gold['Date']=pd.to_datetime(gold['Date'])
gold=gold.set_index('Date')


# In[7]:


gold_new=gold.dropna()


# In[8]:


gold_new.isnull().sum()


# In[9]:


gold_new.dtypes


# In[10]:


sns.pairplot(gold_new,height=1.5)


# In[11]:


gold_new.corr()


# In[12]:


gold_new=gold_new.drop(columns=['Vol.','Change %'])


# In[14]:


# Finding the coefficients of inputs using linear regression

X=gold_new.iloc[:,0:4].values
y1=gold_new.iloc[:,-2].values

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

lin1=LinearRegression()

lin1.fit(X,y1)


# In[15]:


lin1.coef_


# In[17]:


lin1.intercept_


# In[18]:


lin1.score(X,y1)


# In[19]:


# fill the incomplete column 'Pred'

gold['Pred']=lin1.predict(gold.loc[:,['Price','Open','High','Low']].values)

gold.isnull().sum()


# In[20]:


# Fitting the other column using new regression model 

y2=gold_new.iloc[:,-1].values

lin2=LinearRegression()

lin2.fit(X,y2)


# In[21]:


lin2.coef_


# In[22]:


lin2.intercept_


# In[23]:


lin2.score(X,y2)


# In[24]:


# Checking the accuracy of prediction

X_test=gold.iloc[:,0:4].values
y_test=gold.iloc[:,-1]

y2_test_pred=lin2.predict(X_test)

r2_score(y_test,y2_test_pred)

sns.displot(y_test)
sns.displot(y2_test_pred)


# In[25]:


#Linear regression analysis for daily returns

gail=pd.read_csv('GAIL.csv')
gail['Date']=pd.to_datetime(gail['Date'])
gail=gail.set_index('Date')

gail['Series'].value_counts()

nif=pd.read_csv('Nifty50.csv')
nif['Date']=pd.to_datetime(nif['Date'])
nif=nif.set_index('Date')

gail['Daily_return']=gail['Close Price'].pct_change()*100
gail=gail.dropna()

nif['Daily_return']=nif['Close'].pct_change()*100
nif=nif.dropna()


# In[26]:


plt.scatter(nif['Daily_return'],gail['Daily_return'])


# In[27]:


X=nif['Daily_return'].iloc[-1:-91:-1].sort_index()
X=X.values.reshape(-1,1)

y=gail['Daily_return'].iloc[-1:-91:-1].sort_index()
y=y.values


# In[28]:


plt.scatter(X,y)


# In[31]:


from sklearn.linear_model import LinearRegression
lin=LinearRegression()

lin.fit(X,y)


# In[32]:


lin.coef_


# In[33]:


lin.intercept_


# In[34]:


lin.score(X,y)


# In[35]:


y_pred=lin.predict(X)

plt.scatter(X,y)
plt.plot(X,y_pred,color='r')


# In[ ]:


# The Beta value found for daily return is 1.19 which is greater than 1
# Hence we can say that daily return of gail's stocks is more volatile than daily market return by 19% 


# In[36]:


#Linear regression analysis for Monthly returns

gail_month=gail.resample('M').apply(lambda x:x[-1])

nif_month=nif.resample('M').apply(lambda x:x[-1])


gail_month['Monthly_return']=gail_month['Close Price'].pct_change()*100

nif_month['Monthly_return']=nif_month['Close'].pct_change()*100


gail_month=gail_month.dropna()
nif_month=nif_month.dropna()


plt.scatter(nif_month['Monthly_return'],gail_month['Monthly_return'])


# In[37]:


X=nif_month['Monthly_return'].values.reshape(-1,1)
y=gail_month['Monthly_return'].values

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X,y)


# In[38]:


lin.coef_


# In[39]:


lin.intercept_


# In[40]:


lin.score(X,y)


# In[41]:


y_pred=lin.predict(X)

plt.scatter(X,y)
plt.plot(X,y_pred,'r')


# In[ ]:


# The Beta value found for monthly return is 0.712 which is less than 1
# Hence we can say that monthly return of gail's stocks is less volatile than monthly market return by 29%

