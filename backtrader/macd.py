
from mytrade.strategy import MacdCross
import tushare as ts
from  datetime import datetime
import backtrader as bt



ts.set_token("c50898fa9e4e26fa0a46c580d6c45d7e8fc9d19d8607a10f782239c3")
pro = ts.pro_api()
code="000001.SZ"
start='2011-01-01'
end='2021-03-15'
df = pro.query('daily',ts_code=code,start_date=start,end_date=end,autype='qfq')
df.trade_date = df.trade_date.astype('datetime64')
df =df.loc[::-1]
df.reset_index(drop=True, inplace=True)
fromdate = datetime.strptime(start, '%Y-%m-%d') 
todate   = datetime.strptime(end, '%Y-%m-%d') 
data = bt.feeds.PandasData(dataname=df,datetime=1,open=2,high=3,
    low=4,close=5,volume=9,fromdate=fromdate,todate=todate,
    timeframe=bt.TimeFrame.Days)

if __name__ == "__main__":
    from mytrade import cerebro
    cerebro.add_data(data)
    cerebro.add_strategy(MacdCross)
    cerebro.run(plot=True)

