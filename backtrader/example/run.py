
import tushare as ts
from  datetime import datetime
from mytrade.strategy import DoubleAverage,MacdCross,VwapCross
from mytrade.dataset import CustomCsvData
from mytrade import cerebro
#df form tushare 
# ts.set_token("")
# pro = ts.pro_api()
# code="000001.SZ"
# start='2011-01-01'
# end='2021-03-15'
# df = pro.query('daily',ts_code=code,start_date=start,end_date=end,autype='qfq')
# df.trade_date = df.trade_date.astype('datetime64')
# df =df.loc[::-1]
# df.reset_index(drop=True, inplace=True)
# fromdate = datetime.strptime(start, '%Y-%m-%d') 
# todate   = datetime.strptime(end, '%Y-%m-%d') 
# data = bt.feeds.PandasData(dataname=df,datetime=1,open=2,high=3,
#     low=4,close=5,volume=9,fromdate=fromdate,todate=todate,
#     timeframe=bt.TimeFrame.Days)


# #df from csv
#from datetime import datetime
#import pandas as pd
#df = pd.read_csv("binance_BTCUSDT_1min.csv")
#df.open_time = df.open_time.astype('datetime64[ns]')



# data = bt.feeds.PandasData(dataname=df,datetime=0,open=1,high=2,
#     low=3,close=4,volume=5,fromdate=start,todate=end,
#     timeframe=bt.TimeFrame.Ticks)
# data=bt.feeds.GenericCSVData(dataname="binance_BTCUSDT_1min.csv",
#                              fromdate = datetime(2020, 9, 28,12,1),
#                              todate = datetime(2020, 10, 1,23,56),
#                              nullvalue=0.0,
#                              #tmformat=,
#                              dtformat="%Y/%m/%d"+" %H:%M",
#                              datetime=0,
#                              open=1,
#                              high=2,
#                              low=3,
#                              close=4,
#                              volume=5,
#                              timeframe=bt.TimeFrame.Minutes,
#                              rtbar=True,
#                              compression=1, 
#                              openinterest=-1)
data = CustomCsvData(
    timeframe=cerebro.period,
    #from 2018-9-20
    fromdate=datetime(2018,9, 20),
    todate=datetime(2019, 1,1 ),
    compression=1
)
if __name__ == "__main__":

    #cerebro.resampledata(data, timeframe=cerebro.period, compression=10)
    cerebro.add_data(data)
    cerebro.add_strategy(VwapCross)
    cerebro.run(plot=1)
    print("finally property:{}.".format(cerebro.broker.getvalue()))

