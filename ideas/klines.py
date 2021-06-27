from pandas import DataFrame, Series
from finta import TA
from ta.volume import VolumeWeightedAveragePrice as VWAP


class KlineCreator(object):
    """docstring for KlineCeater"""
    def __init__(self,ohlc,symbol,period,passover=200):
        self._ohlc  = ohlc
        self.symbol = symbol
        self.period = period
        self.activate_ohlc = ohlc[:passover]
        self.step   = passover
        self.activable = True
        self.acti_ohlc_len = ohlc[:passover].shape[0]

    def get_current_price(self,attention="close"):
        price = self.activate_ohlc[-1:][attention][0]
        return price

    def update(self):
        if self.acti_ohlc_len == self._ohlc.shape[0]:
            self.activable = False
            assert self.activable is True,"activate_ohlc ending with unactivable."
        else:
            self.activate_ohlc = self.activate_ohlc.append(self._ohlc[self.step:self.step + 1])
            self.step += 1
            self.acti_ohlc_len += 1
    
    def forward(self,step):
        assert self.step + step < self._ohlc.shape[0],"No many step to the  activate ohlc ending." 
        for i in range(step):
            self.update()


    def ichimoku(self, t1=9, t2=26, t3=52):
        _t1_max = Series.rolling(self.activate_ohlc.high, t1).max() 
        _t1_min = Series.rolling(self.activate_ohlc.low, t1).min()
        _t2_max = Series.rolling(self.activate_ohlc.high, t2).max() 
        _t2_min = Series.rolling(self.activate_ohlc.low, t2).min()
        _t3_high_Average = Series.rolling(self.activate_ohlc.high, t3).mean()
        _t3_low_Average  = Series.rolling(self.activate_ohlc.low, t3).mean()
        Tenkan_Sen = (_t1_max + _t1_min )/ 2
        Kijun_Sen = (_t2_max + _t2_min) / 2
        Senkou_Span_A = (Tenkan_Sen + Kijun_Sen) / 2
        Senkou_Span_B = (_t3_high_Average + _t3_low_Average) / 2
        df = DataFrame(dict(Tenkan_Sen=Tenkan_Sen, Kijun_Sen=Kijun_Sen,
                                   Senkou_Span_A=Senkou_Span_A.shift(t2),
                                   Senkou_Span_B=Senkou_Span_B.shift(t2),
                                   Chikou_Span=self.activate_ohlc.close.shift(-t2)))
        name = "ichimoku"
        return df
    
    def macd(self,fast=12, slow=26, signal=9):
        df = TA.MACD(self.activate_ohlc,period_fast=fast,period_slow=slow,signal=signal)
        return df

    def trix(self,period=120):
        df = TA.TRIX(self.activate_ohlc,period=60)
        return df

    def rsi(self,period=14,column="close"):
        df = TA.RSI(self.activate_ohlc,period=period,column=column)
        return df
    
    def wto(self):
        df = TA.WTO(self.activate_ohlc)
        return df
    
    def dc(self,period=60):
        df = TA.DO(self.activate_ohlc,upper_period=period,lower_period=period)
        return df

    def dmi(self,period=14):
        df = TA.DMI(self.activate_ohlc,period=period)
        return df
        
    def roc(self,period=20):
        df = TA.ROC(self.activate_ohlc,period=period)
        return df

    def vwap(self,period=1):
        _result = VWAP(high=self.activate_ohlc.high,low=self.activate_ohlc.low,close=self.activate_ohlc.close,volume=self.activate_ohlc.volume,window=period)
        df = _result.vwap
        return df
        
    def wma(self,period=9):
        df = TA.WMA(period=9,column="close")
        return df
        
    # def vwma_x(self,period=20):
    #     _v_p = self.activate_ohlc.volume[-period:]
    #     _c_p = self.activate_ohlc.close [-period:]
    #     _v_mul_c =  _v_p.mul(_c_p)
    #     value = _v_mul_c.sum() / _v_p.sum()  
    #     return value 
  
    def vwma(self,period=20):
        _len = self.acti_ohlc_len
        if _len < period:
            _data = Series(index=self.activate_ohlc[:_len].index,dtype="float64")
        else:
            _data = Series(index=self.activate_ohlc[:period -1].index,dtype="float64")
            for i in range(_len - period + 1):
                _v_p = self.activate_ohlc.volume[i : i + period]
                _c_p = self.activate_ohlc.close [i : i + period]
                _v_mul_c =  _v_p.mul(_c_p)
                _vwma = _v_mul_c.sum() / _v_p.sum()
                _value = Series(_vwma,index=self.activate_ohlc[i - 1 + period : i + period].index)
                _data = _data.append(_value)
        return _data
