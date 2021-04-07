import pandas as pd

from carrier.starter import starter 
starter.initialize("config.json")

from carrier.tasks import LoopRunTask,SingleTask
from carrier.utils import logger
from carrier.trade.trade import huobi_usdt_swap_trade as hust
from carrier.market.market import  huobi_usdt_swap_market as husm
from carrier.market.market import  huobi_usdt_spot_market as huspm
from collections import deque
from carrier.utils.tools import get_cur_timestamp,ts_to_datetime_str


class klineCreator:
    def __init__(self,**kwargs):
        self.task = LoopRunTask
        self.data_lenght = kwargs.get("data_lenght",300)
        self.conts_max   = kwargs.get("conts_max",1000)
        self.data = dict()
        self.conts_data  = dict()
        self.cur_ts = None
        self.init_done = None


    async def found_kline(self, symbol, period, size , callback, **kwargs):
        """
            data:{
                huobi_usdt_spot_btc_usdt_1min :deque[{...},...]
            }
        """
        tag = "{}_{}_{}".format(callback.platform,symbol,period)
        # import pdb
        # pdb.set_trace()
        await callback.get_klines(symbol, period, **{"size":size})
        kline = callback.history.get(tag)
        if kline and len(kline) > 0:
            if self._check_kline_ts_continuous(kline):
                logger.info("Kline datetime continuous!",caller=self)
                self.data[tag] = deque(maxlen=self.data_lenght)
                self.data[tag].extend(kline)
                df = self.dataframe(kline)
                df.to_csv("{}.csv".format(tag))
                task_id = kwargs.get("task_id")
                self.task.unregister(task_id)
            else:
                logger.error("Kline datetime discontinuous!",caller=self)

    def dataframe(self,data):
        _df = pd.DataFrame(data)    
        # _df  =  _df.iloc[::-1]
        _df["datetime"] = pd.to_datetime(_df["id"],unit="s")
        _df["volume"] = _df["vol"]
        df = _df[["datetime","open","high","low","close","volume","id"]]
        df = df.set_index("datetime")
        return df

    async def update_ts(self, **kwargs):
        self.cur_ts = get_cur_timestamp() 

    def _check_kline_ts_continuous(self,kline):
        for i in range(len(kline) - 2):
            if (kline[i]["id"] - kline[i + 1]["id"]) == (kline[i+1]["id"] - kline[i+2]["id"]):
                return True
            return   

    def write_f(self,tag,data):
        if not hasattr(self,"{}_f".format(tag)):
            setattr(self,"{}_f".format(tag),open("{}.txt".format(tag),"w"))
        else:
            writer = getattr(self,"{}_f".format(tag))
            writer.write("{},{}\n".format(ts_to_datetime_str(data["id"]),data["close"]))
            writer.flush()


    def run(self, **kwargs):
        self.task.register(self.found_kline, 5,period="1min" ,size=self.data_lenght, symbol="eth_usdt", callback=husm,**kwargs)
        self.task.register(self.found_kline, 5,period="1min" ,size=self.data_lenght, symbol="btc_usdt", callback=huspm,**kwargs)
        self.task.register(self.update_ts, 1, **kwargs)


kc = klineCreator(**{"data_lenght":2000,"conts_max":1000})
kc.run()

starter.start()