import json
import pandas as pd

from klines import KlineCreator



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
        ohlc = ohlc.set_index("date")
    else:
        _df["datetime"] = pd.to_datetime(_df["id"],unit="s")
        _df["volume"] = _df["vol"]
        ohlc = _df[["datetime","open","high","low","close","volume"]]
        ohlc = ohlc.set_index("datetime")
        
    kc = KlineCreator(symbol=symbol,period=period,ohlc=ohlc)
    return kc


