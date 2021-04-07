from pandas_datareader.data import DataReader
import pandas as pd
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
style. use('ggplot')


from matplotlib import style
style. use('ggplot')
import time
day = dt.datetime.utcfromtimestamp(time.time()).day
data  = pd.read_csv('huobi_usdt_spot_btc_usdt_1min_{}.txt'.format(day))
prices  = data.iloc[-500::,1]
returns   = prices.pct_change()
last_price  = prices.values[-1]


start = dt.datetime(2020,3,9)
end = dt.datetime(2021,3,9)
prices = DataReader("XLM-USD","yahoo",start,end)["Close"]
returns =  prices.pct_change()
last_price = prices[-1]

number_simulations = 1000
number_days =  len(prices.index)
simulation_df = pd.DataFrame()
daily_volatility = returns.std()

for x in range(number_simulations):
    count = 0
    price_series = []
    
    price = last_price * (1+np.random.normal(0,daily_volatility))
    price_series.append(price)

    for y in range(number_days):
        if count == number_days-1:
            break
        price = price_series[count] * (1 + np.random.normal(0,daily_volatility))
        price_series.append(price)
        count += 1
    simulation_df[x] = price_series


fig = plt.figure(figsize=(15,9))
plt.plot(simulation_df)
plt.axhline(y=last_price,color="r",linestyle="-")
plt.title("Monte Carlo Simulation: BTC-USD",fontsize=34)
plt.xlabel('day')
plt.ylabel('price')
#plt.show()#Monte Carlo Simulation : BTC-USD10000


#check mean median max min of enjoy_values
eoy_values = simulation_df.iloc[number_days-1].sort_values(ascending=False) 

import seaborn as seabornInstance
import seaborn as sns
#plt.style.use('seaborn-poster')
#%matplotlib inline
seabornInstance.displot(eoy_values,color="darkorange")
# plt.axhline(eoy_values.median())
# plt.axhline(eoy_values.max())
# plt.axhline(eoy_values.min())
plt.show()
