#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('week3.csv')


# In[3]:


df.dtypes


# In[4]:


df['Date']=pd.to_datetime(df['Date'])


# In[5]:


data=df.loc[13:,['Close Price','MA','UP_band','LO_band']]


# In[6]:


data=data.set_index(np.arange(len(data)))


# In[7]:


data


# In[8]:


l=[]
for i in range(len(data['Close Price'])):
    if(data['Close Price'][i]<data['LO_band'][i]):
        l.append('Buy')
    elif(data['Close Price'][i]>=data['LO_band'][i] and data['Close Price'][i]<data['MA'][i]):
        l.append('Hold Buy')
    elif(data['Close Price'][i]>=data['MA'][i] and data['Close Price'][i]<data['UP_band'][i]):
        l.append('Hold Short')
    else:
        l.append('Short')


# In[9]:


data['Call']=pd.Series(l)    


# In[10]:


data


# In[11]:


X=data.loc[:,['Close Price','MA','LO_band','UP_band']]

y=data.iloc[:,-1]


# In[12]:


from sklearn.preprocessing import LabelEncoder

lab=LabelEncoder()

y=lab.fit_transform(y)


# In[13]:


label=lab.classes_
label


# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)


from sklearn.linear_model import LogisticRegression
log=LogisticRegression(max_iter=5000)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=8)

from sklearn.ensemble import RandomForestClassifier
rand=RandomForestClassifier()


from sklearn.svm import SVC
svm=SVC()


# In[15]:


# Logistic Regression


# In[16]:


log.fit(X_train,y_train)

log.score(X_train,y_train)


# In[17]:


log.score(X_test,y_test)


# In[18]:


log.predict(X_test)


# In[19]:


# Decision tree


# In[20]:


dt.fit(X_train,y_train)

dt.score(X_train,y_train)


# In[21]:


dt.score(X_test,y_test)


# In[22]:


# RandomForest 


# In[23]:


rand.fit(X_train,y_train)

rand.score(X_train,y_train)


# In[24]:


rand.score(X_test,y_test)


# In[25]:


# Support Vector Machine


# In[26]:


svm.fit(X_train,y_train)

svm.score(X_test,y_test)


# In[27]:


# importing Gail dataset and predicting the daily calls 


# In[28]:


gail=pd.read_csv('GAIL.csv')

ma=gail['Close Price'].rolling(14).mean()
std=gail['Close Price'].rolling(14).std()
lo=ma-2*std
up=ma+2*std


# In[29]:


data1=pd.DataFrame({'Stock_Price':gail['Close Price'],
              'MA':ma,'LO_band':lo,'UP_band':up})


# In[30]:


data1


# In[31]:


data1=data1.dropna()


# In[32]:


data1


# In[33]:


label_pred=log.predict(data1)


# In[34]:


calls=[]

for e in label_pred:
    if(e==0):
        calls.append('Buy')
    elif(e==1):
        calls.append('Hold Buy')
    elif(e==2):
        calls.append('Hold Short')
    else:
        calls.append('Short')


# In[40]:


data1=data1.assign(Calls=calls)


# In[41]:


data1


# In[42]:


data1['Calls'].value_counts()


# In[43]:


# importing the dataset


# In[44]:


tcs=pd.read_csv('TCS.csv')


# In[45]:


tcs.columns


# In[46]:


tcs['%OC']=tcs.loc[:,['Open Price','Close Price']].diff(axis=1)['Close Price']/tcs['Open Price']*100
tcs['%LH']=tcs.loc[:,['Low Price','High Price']].diff(axis=1)['High Price']/tcs['Low Price']*100

tcs['roll_mean']=tcs['Close Price'].pct_change().rolling(5).mean()*100
tcs['roll_std']=tcs['Close Price'].pct_change().rolling(5).std()*100


# In[47]:


l2=[]
for i in range(len(tcs['Close Price'])):
    if(tcs['Close Price'][i+1]>tcs['Close Price'][i]):
        l2.append(1)
    elif(tcs['Close Price'][i+1]<tcs['Close Price'][i]):
        l2.append(-1)    
    else:
        break


# In[48]:


tcs['Action']=pd.Series(l2)


# In[49]:


tcs=tcs.dropna()


# In[50]:


X=tcs.loc[:,['%OC','%LH','roll_mean','roll_std']].values
y=tcs.iloc[:,-1]


# In[51]:


from sklearn.ensemble import RandomForestClassifier
rand=RandomForestClassifier()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)


# In[52]:


rand.fit(X_train,y_train)

rand.score(X_train,y_train)


# In[53]:


rand.score(X_test,y_test)


# In[54]:


# plot of cumulative return 
plt.figure(figsize=(15,8))
tcs['cum_return']=(tcs['Close Price'].pct_change()+1).cumprod()
plt.plot(tcs['cum_return'])

