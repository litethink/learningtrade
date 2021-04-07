#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('JUBLFOOD.csv')


# In[3]:


data['Series'].value_counts()


# In[4]:


data=data[data.Series!='BL']


# In[6]:


# analyzing data 


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.describe()


# In[13]:


# computing min,max,mean value of Close price for last 90 days  

Close_Price_min=data['Close Price'].iloc[-1:-91:-1].min()
Close_Price_max=data['Close Price'].iloc[-1:-91:-1].max()
Close_price_mean=data['Close Price'].iloc[-1:-91:-1].mean()
print('Close_Price_min',Close_Price_min)
print('Close_Price_max',Close_Price_max)
print('Close_Price_meax',Close_price_mean)


# In[14]:


# analyzing data types of each column

data.dtypes


# In[15]:


# converting datatype of date column from object to datetime64[ns]

data['Date']=pd.to_datetime(data['Date'])


# In[16]:


# creating dataframe of month and year 

data['month']=pd.DatetimeIndex(data['Date']).month
data['year']=pd.DatetimeIndex(data['Date']).year

data['Date'].max()-data['Date'].min()


# In[17]:


# grouping of dataframe by year and month

gp=data.groupby(['year','month'])
for m,d in gp:
    print(*m)
    print(d)


# In[32]:


# creating vwap function 
   
def vwap_func(df):
    cum_sum=(((df['High Price']+df['Low Price']+df['Close Price'])/3)*df['Total Traded Quantity']).cumsum()
    cum_vol=df['Total Traded Quantity'].cumsum()
    return cum_sum/cum_vol


# In[33]:


# computing vwap values for each month

vwap=gp.apply(vwap_func)
vwap


# In[34]:


# creating function for average stock price for last N days 

def avg_price(n):
    avg=data['Close Price'].iloc[-(n+1):-1:1].mean()
    return avg


# In[35]:


# creating function for profit/loss percentage of stock price between last N days

def perc_ch(n):
    perc=((data['Close Price'].iloc[-1]-data['Close Price'].iloc[-(n+1)])/data['Close Price'].iloc[-1])*100
    return perc


# In[40]:


# computing average stock price for last 1 week,2 week,1 month,3 months,6 months,1 year
print('average stock price for last 1 week',avg_price(7))
print('average stock price for last 2 week',avg_price(14))
print('average stock price for last 1 month',avg_price(30))
print('average stock price for last 3 month',avg_price(91))
print('average stock price for last 6 month',avg_price(182))
print('average stock price for last 1 year',avg_price(364))


# In[47]:


# computing profit/loss percentage of stock price for last 1 week,2 week,1 month,3 months,6 months,1 year

print('profit/loss percentage of stock price for last 1 week',perc_ch(7))
print('profit/loss percentage of stock price for last 2 week',perc_ch(14))
print('profit/loss percentage of stock price for last 1 month',perc_ch(30))
print('profit/loss percentage of stock price for last 3 months',perc_ch(91))
print('profit/loss percentage of stock price for last 6 months',perc_ch(182))
print('profit/loss percentage of stock price for last 1 year',perc_ch(364))


# In[43]:


# adding a column for daily change in percentage of closing price

data['Day_Perc_Change']=data['Close Price'].pct_change()*100
data['Day_Perc_Change']=data['Day_Perc_Change'].fillna(0)
data.head()


# In[45]:


# adding a column of Trend

condtion_list=[data['Day_Perc_Change']<=-7.0,np.logical_and(data['Day_Perc_Change']>-7.0,data['Day_Perc_Change']<=-3.0),
               np.logical_and(data['Day_Perc_Change']>-3.0,data['Day_Perc_Change']<=-1.0),np.logical_and(data['Day_Perc_Change']>-1.0,data['Day_Perc_Change']<=-0.5),
               np.logical_and(data['Day_Perc_Change']>-0.5,data['Day_Perc_Change']<=0.5),np.logical_and(data['Day_Perc_Change']>0.5,data['Day_Perc_Change']<=1.0),
               np.logical_and(data['Day_Perc_Change']>1.0,data['Day_Perc_Change']<=3.0),np.logical_and(data['Day_Perc_Change']>3.0,data['Day_Perc_Change']<=7.0),
               data['Day_Perc_Change']>7.0]

choice_list=['Bear Drop','Among top Losers','Negative','Slight Negative','Slight or No Change','Slight Positive',
            'Positive','Among Top gainers','Bull run']

data['Trend']=np.select(condtion_list,choice_list)
data.head()


# In[46]:


# computing the mean and median of Total Traded Quantity for each type of Trend

grp=data['Total Traded Quantity'].groupby(data['Trend'])
grp.agg(['mean','median'])


# In[ ]:


# saving the dataframe with additional columns

data.to_csv('week2.csv')

