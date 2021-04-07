import pandas as pd

from carrier.starter import starter 
starter.initialize("../config.json")

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

    async def update_kline(self, symbol, period, callback, **kwargs):
        """
            {'period': '1min', 'size': 2, 'symbol': 'btc_usdt',
             'task_id': '47ae4272-7ee6-11eb-a2d8-4bea7b2049e6', 'heart_beat_count': 200}

         """
        tag = "{}_{}_{}".format(callback.platform,symbol,period)
        if not self.data.get(tag):
            return 
        if not self.conts_data.get(tag):
            self.conts_data[tag] = deque(maxlen=self.conts_max)
        kline = await callback.get_klines(symbol,period,size=2)
        if kline:
            _new_data = callback.history.get(tag)  
            self.conts_data[tag].append(_new_data[-1])
            # import pdb
            # pdb.set_trace()
            data =  self.data[tag][-3],self.data[tag][-2],self.data[tag][-1]
            if self.cur_ts - data[-1]["id"] > 3 * 60:
                task_id = kwargs.get("task_id")
                logger.error("updating kline but time out!",caller=self)
                getattr(self,"{}_f".format(tag)).close()
                self.task.unregister(task_id)
            if self._check_update_ts_continuous(data):
                if data[-1]["id"] == _new_data[-1]["id"]:
                    self.data[tag].pop()
                    self.data[tag].append(_new_data[-1])
                if data[-1]["id"] == _new_data[0]["id"]:
                    self.data[tag].pop()
                    self.write_f(tag,_new_data[0])
                    self.data[tag].extend(_new_data)
                logger.info(self.data,caller=self)
            else:
                task_id = kwargs.get("task_id")
                getattr(self,"{}_f".format(tag)).close()
                logger.error("updating kline  but datetime discontinuous!",caller=self)
                import pdb
                pdb.set_trace()
                self.task.unregister(task_id)


    async def found_kline(self, symbol, period, size , callback, **kwargs):
        """
            data:{
                huobi_usdt_spot_btc_usdt_1min :deque[{...},...]
            }
        """
        tag = "{}_{}_{}".format(callback.platform,symbol,period)
        setattr(self,"{}_f".format(tag),open("{}.txt".format(tag),"w"))
        # import pdb
        # pdb.set_trace()
        await callback.get_klines(symbol, period, **{"size":size})
        kline = callback.history.get(tag)
        if kline and len(kline) > 0:
            if self._check_kline_ts_continuous(kline):
                logger.info("Kline datetime continuous!",caller=self)
                self.data[tag] = deque(maxlen=self.data_lenght)
                self.data[tag].extend(kline)
                for data in kline[:-1]:
                    self.write_f(tag,data)
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
        writer = getattr(self,"{}_f".format(tag))
        writer.write("{},{}\n".format(ts_to_datetime_str(data["id"]),data["close"]))
        writer.flush()

    def _check_update_ts_continuous(self,data):
        # import pdb
        # pdb.set_trace()
        if abs(data[-3]["id"] - data[-2]["id"]) == 60 and abs(data[-2]["id"] - data[-1]["id"]) == 60:
            return True
        return    

    def run(self, **kwargs):
        self.task.register(self.found_kline, 5,period="1min" ,size=self.data_lenght, symbol="eth_usdt", callback=husm,**kwargs)
        self.task.register(self.found_kline, 5,period="1min" ,size=self.data_lenght, symbol="btc_usdt", callback=huspm,**kwargs)
        self.task.register(self.update_ts, 1, **kwargs)
        self.task.register(self.update_kline, 1, period="1min" , symbol="eth_usdt",callback=husm,**kwargs)
        self.task.register(self.update_kline, 1, period="1min" , symbol="btc_usdt",callback=huspm,**kwargs)


kc = klineCreator(**{"data_lenght":2000,"conts_max":1000})
kc.run()

starter.start()
