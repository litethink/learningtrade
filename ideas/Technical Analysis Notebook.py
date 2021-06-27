#!/usr/bin/env python
# coding: utf-8

# In[3]:

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
get_ipython().run_line_magic('matplotlib', 'inline')


def create_klines():
    with open("data.json","r",encoding="utf-8") as f:
        data = json.load(f)

    _1,symbol,_2,period = data["ch"].split(".")
    _df = pd.DataFrame(data["data"])
    _df  =  _df.iloc[::-1]
    if period == "1day":
        _df["date"] = pd.to_datetime(_df["id"],unit="s")
        _df["date"] = _df["date"].dt.date
        _df["volume"] = _df["vol"]
        ohlc = _df[["date","open","high","low","close","volume"]]
    else:
        _df["datetime"] = pd.to_datetime(_df["id"],unit="s")
        _df["volume"] = _df["vol"]
        ohlc = _df[["datetime","open","high","low","close","volume"]]
    return ohlc



# In[22]:


def RSI(dataframe, period):
    '''
    Computes the RSI of a given price series for a given period length
    :param dataframe:
    :param period:
    :return dataframe with rsi:
    '''

    rsi = []

    for stock in dataframe['symbol'].unique():
        all_prices = dataframe[dataframe['symbol'] == stock]['close']
        diff = np.diff(all_prices) # length is 1 less than the all_prices
        for i in range(period):
            rsi.append(None) # because RSI can't be calculated until period prices have occured

        for i in range(len(diff) - period + 1):
            avg_gain = diff[i:period + i]
            avg_loss = diff[i:period + i]
            avg_gain = abs(sum(avg_gain[avg_gain >= 0]) / period)
            avg_loss = abs(sum(avg_loss[avg_loss < 0]) / period)
            if avg_loss == 0:
                rsi.append(100)
            elif avg_gain == 0:
                rsi.append(0)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))

    dataframe['RSI'] = rsi
    return dataframe


def PROC(dataframe, period):
    '''
    Computes the PROC(price rate of change) of a given price series for a given period length
    :param dataframe:
    :param period:
    :return proc:
    '''

    proc = []

    for stock in dataframe['symbol'].unique():
        all_prices = list(dataframe[dataframe['symbol'] == stock]['close'])
        for i in range(period):
            proc.append(None) # because proc can't be calculated until period prices have occured
        for i in range(len(all_prices) - period):
            if len(all_prices) <= period:
                proc.append(None)
            else:
                try:
                    calculated = (all_prices[i + period] - all_prices[i]) / all_prices[i]
                    proc.append(calculated)
                except:
                    import pdb 
                    pdb.set_trace()
    dataframe['PROC'] = proc
    return dataframe


def SO(dataframe, period):
    
    so = []
    
    for stock in dataframe['symbol'].unique():
        all_prices = list(dataframe[dataframe['symbol'] == stock]['close'])
        
        for i in range(period):
            so.append(None)
 
        for i in range(len(all_prices) - period):
            C = all_prices[i]
            H = max(all_prices[i:i+period])
            L = min(all_prices[i:i+period])
            so.append(100 * ((C - L) / (H - L)))

    dataframe['SO'] = so
    return dataframe


def Williams_R(dataframe, period):
    '''
    Williams %R
    Calculates fancy shit for late usage. Nice!

    EXAMPLE USAGE:
    data = pandas.read_csv("./data/ALL.csv", sep=",",header=0,quotechar='"')
    wr = Williams_R(data)
    print(wr)

    '''
    
    wr = []
    
    for stock in dataframe['symbol'].unique():
        all_prices = list(dataframe[dataframe['symbol'] == stock]['close'])
        for i in range(period):
            wr.append(None) # because proc can't be calculated until period prices have occured
            
        for i in range(period-1,len(all_prices)-1):
            C = all_prices[i]
            H = max(all_prices[i-period+1:i])
            L = min(all_prices[i-period+1:i])
            wr_one = (
                ((H - C) 
                 / (H - L)) * -100
            )
            if wr_one <=-100:
                wr.append(-100)
            elif wr_one >= 100:
                wr.append(100)
            else:
                wr.append(wr_one)
    dataframe["WR"] = wr
    return dataframe


def calculate_targets(df, period):
    
    targets = []

    for stock in df['symbol'].unique():
        all_prices = list(df[df['symbol'] == stock]['close'])
        
        for i in range(0, len(all_prices)-period):
            targets.append(np.sign(all_prices[i+period] - all_prices[i]))
        for i in range(len(all_prices)-period, len(all_prices)):
            targets.append(None)

    df["Target({})".format(period)] = targets
    return df


def On_Balance_Volume(dataframe):
    '''
    Williams %R
    Calculates fancy shit for late usage. Nice!

    EXAMPLE USAGE:
    data = pandas.read_csv("./data/ALL.csv", sep=",",header=0,quotechar='"')
    wr = Williams_R(data)
    print(wr)

    '''
    obv = []
    
    for stock in dataframe['symbol'].unique():
        all_prices = list(dataframe[dataframe['symbol'] == stock]['close'])
        all_volumes = list(dataframe[dataframe['symbol'] == stock]['volume'])
    
        obv.append(dataframe.iloc[0]["volume"])
        for i in range(1,len(all_prices)):
            C_old = all_prices[i-1]
            C = all_prices[i]
            if(C > C_old):
                obv.append(obv[i-1]+ all_volumes[i])
            elif (C < C_old):
                obv.append(obv[i - 1] - all_volumes[i])
            else:
                obv.append(obv[i-1])

    dataframe['OBV'] = obv
    return dataframe


def delete_bad_data(df):
    for stock in df['symbol'].unique():
        if not df[df["symbol"]==stock]["close"] .all():
            df = df.drop(df[df["symbol"]==stock]["close"].index,axis=0)
        if not df[df["symbol"]==stock]["volume"] .all():
            df = df.drop(df[df["symbol"]==stock]["volume"].index,axis=0)
        if not df[df["symbol"]==stock]["open"] .all():
            df = df.drop(df[df["symbol"]==stock]["open"].index,axis=0)
    return df


#def delete_bad_data(df):
#    for stock in df['symbol'].unique():
#        if df[df["symbol"]==stock]["close"].any() < 1:
#            df = df.drop(df[df["symbol"]==stock], axis=0)
#        if df[df["symbol"]==stock]["volume"].any() == 0:
#           df = df.drop(df[df["symbol"]==stock], axis=0)
#        if df[df["symbol"]==stock]["open"].any() == None:
#           df = df.drop(df[df["symbol"]==stock], axis=0)
#    return df




data = create_klines()
data["symbol"] = "btc"
data = delete_bad_data(data)

rsi1 = RSI(data,14)[["RSI"]].values[-2][0]
data = data.drop("RSI",axis=1)

rsi2 = RSI(data[:-1],14)[["RSI"]].values[-1][0]

if rsi1 == rsi2:
    data = RSI(data,14)
    print("RSI: Done")


# In[ ]:
proc1 = PROC(data,14)[["PROC"]].values[-2][0]
data = data.drop("PROC",axis=1)
proc2 = PROC(data[:-1],14)[["PROC"]].values[-1][0]
if proc1 == proc2:
    data = PROC(data, 14)
    print("PROC: Done")


# In[ ]:

so1 = SO(data,14)[["SO"]].values[-2][0] 
data = data.drop("SO",axis=1)
so2 = SO(data[:-1],14)[["SO"]].values[-1][0]
if so2 == so1:
    data = SO(data,14)
    print("SO: Done")


# In[ ]:
wr1 = Williams_R(data,14)[["WR"]].values[-2][0]
data = data.drop("WR",axis=1)
wr2 = Williams_R(data[:-1],14)[["WR"]].values[-1][0]
if wr1 == wr2:
    data = Williams_R(data, 14 )
    print("Williams_R: Done")


obv1 = On_Balance_Volume(data)[["OBV"]].values[-2][0]
data = data.drop("OBV",axis=1)
obv2 = On_Balance_Volume(data[:-1])[["OBV"]].values[-1][0]
if obv2 == obv1:
    data = On_Balance_Volume(data)
    print("On_Balance_Volume: Done")

ema1 =  pd.DataFrame.ewm(data[:-1]["close"], com=.5).mean().values[-1]
ema2 = pd.DataFrame.ewm(data["close"], com=.5).mean().values[-2]
if ema2 == ema1:
    data["EWMA"] = pd.DataFrame.ewm(data["close"], com=.5).mean()
    print("EWMA: Done")



def detrend_data(df):
    trend = None
    for stock in df['symbol'].unique():
        all_prices = list(df[df['symbol'] == stock]['close'])
#        trend.append(signal.detrend(all_prices))
        if(trend is None):
            trend = list(signal.detrend(all_prices))
        else:
            trend.extend(signal.detrend(all_prices))
        print("len(trend):{} len(df['symbol']):{}".format(len(trend),len(all_prices)))

    print("len(trend):{} len(df):{}".format(len(trend),len(df)))
    df['close_detrend'] = trend
    return df


dd1 = detrend_data(data)["close_detrend"].values[-2]
data = data.drop("close_detrend",axis=1)
dd2    = detrend_data(data[:-1])["close_detrend"].values[-1]
#dd1 != dd2
if dd1 == dd2:
    detrend_data(data)
    print("Date detrend: Done")

#use detrend_data
# selected_data = data[:1000]
# detrend_data(selected_data)
# selected_data = selected_data.dropna(axis=0, how='any')
# prediction_data = data[1000:]
# In[ ]:
# full_data = selected_data
# l = []
# for i in range(len(prediction_data)):
#     a = prediction_data[i:i+1]
#     full_data = full_data.append(a)
#     detrend_data(full_data)
#     l.append(full_data["close_detrend"].values[-1])
    
# prediction_data["close_detrend"] = l

#use obv
selected_data = data[:1000]
selected_data = selected_data.dropna(axis=0, how='any')
prediction_data = data[1000:]
process detrend data



selected_data = calculate_targets(selected_data, 1)
selected_data = calculate_targets(selected_data, 3)
selected_data = calculate_targets(selected_data, 5)
selected_data = calculate_targets(selected_data, 10)
selected_data = calculate_targets(selected_data, 14)
selected_data = calculate_targets(selected_data, 30)
print('Targets Done - except 60')
selected_data = selected_data.dropna(axis=0, how='any')

prediction_data = prediction_data.reset_index()
# In[ ]:


selected_data=selected_data.reset_index()
 


# In[ ]:


selected_data.to_csv("./selected_data.csv")
prediction_data.to_csv("./prediction_data.csv")





