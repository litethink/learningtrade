#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('train/huobi_usdt_spot_btc_usdt_1min.csv')
train_df['datetime'] = pd.to_datetime(train_df['datetime'])

ma  = train_df['close'].rolling(14).mean()
std = train_df['close'].rolling(14).std()
lo  = ma - 2 * std
up  = ma + 2 * std

train_data = pd.DataFrame({'Close':train_df['close'],'MA':ma,'Low_band':lo,'Up_band':up})


l=[]
for i in range(len(train_data['Close'])):
    if train_data['Close'].values[i] < train_data['Low_band'].values[i]:
        l.append('Long')
    elif train_data['Close'].values[i] >= train_data['Low_band'].values[i]  and train_data['Close'].values[i] < train_data['MA'].values[i]:
        l.append('Hold long')
    elif train_data['Close'].values[i] >= train_data['MA'].values[i]  and train_data['Close'].values[i] < train_data['Up_band'].values[i]:
        l.append('Hold short')
    else:
        l.append('Short')

train_data['Call'] = pd.Series(l)    
train_data = train_data.dropna()

X = train_data.loc[:,['Close','MA','Low_band','Up_band']]
y = train_data.iloc[:,-1]
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
y = lab.fit_transform(y)
label = lab.classes_
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


from sklearn.linear_model import LogisticRegression

log = LogisticRegression(max_iter=5000)
log.fit(X_train,y_train)
log.score(X_train,y_train)
log.score(X_test,y_test)
log.predict(X_test)





real_df = pd.read_csv('train/huobi_usdt_spot_btc_usdt_1min.csv')
real_df['datetime'] = pd.to_datetime(real_df['datetime'])

ma  = real_df['close'].rolling(14).mean()
std = real_df['close'].rolling(14).std()
lo  = ma - 2 * std
up  = ma + 2 * std

real_data  = pd.DataFrame({'Close':real_df['close'],'MA':ma,'Low_band':lo,'Up_band':up})
real_data  = real_data.dropna()
label_pred = log.predict(real_data)


# In[34]:

label_dict = {
    0 : "Hold long",
    1 : "Hold short",
    2 : "Long",
    3 : "Short"
}

calls = []
for e in label_pred:
    calls.append(label_dict[e])

real_data=real_data.assign(Calls=calls)



#ramdomly forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
train_df['%OC'] = train_df.loc[:,['open','close']].diff(axis=1)['close']/train_df['open']*100
train_df['%LH'] = train_df.loc[:,['low','high']].diff(axis=1)['high']/train_df['low']*100
train_df['roll_mean'] = train_df['close'].pct_change().rolling(5).mean()*100
train_df['roll_std']  = train_df['close'].pct_change().rolling(5).std()*100

real_df['%OC'] = real_df.loc[:,['open','close']].diff(axis=1)['close']/real_df['open']*100
real_df['%LH'] = real_df.loc[:,['low','high']].diff(axis=1)['high']/real_df['low']*100
real_df['roll_mean'] = real_df['close'].pct_change().rolling(5).mean()*100
real_df['roll_std']  = real_df['close'].pct_change().rolling(5).std()*100


l2 = []
for i in range(len(train_df['close'])):
    if(train_df['close'][i+1]>train_df['close'][i]):
        l2.append(1)
    elif(train_df['close'][i+1]<train_df['close'][i]):
        l2.append(-1)    
    else:
        break

train_df['Action'] = pd.Series(l2)
train_df = train_df.dropna()
X = train_df[['%OC','%LH','roll_mean','roll_std']].values
y = train_df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y)
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
rf.score(X_test,y_test)
# plot of cumulative return 
plt.figure(figsize = (15,8))
train_df['cum_return'] = (train_df['close'].pct_change()+1).cumprod()
plt.plot(train_df['cum_return'])



real_df = real_df.dropna()
real_X = real_df[['%OC','%LH','roll_mean','roll_std']].values
real_y = rf.predict(real_X)
real_y = pd.Series(real_y,dtype="int8")
real_df["Action"] = real_y
# plot of cumulative return 
plt.figure(figsize = (15,8))
real_df['cum_return'] = (real_df['close'].pct_change()+1).cumprod()
plt.plot(real_df['cum_return'])

