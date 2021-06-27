from product import create_klines
from trade import TradeOperator
kc = create_klines()
to = TradeOperator(capital=10000)
to.init_symbol(kc.symbol,current_price=kc.get_current_price())
df = kc._ohlc



class IndexState(object):
    """docstring for IndexState"""
    def __init__(self):
        self.macd_slow_over_fast = None
        self.macd_fast_over_slow = None    
        self.dc_bottom_slope_positive = None
        self.dc_bottom_slope_negative = None
        self.dc_top_slope_positive    = None
        self.dc_top_slope_negative  = None
        self.trix_value_negative    = None
        self.trix_value_positive    = None
        self.trix_slope_negative    = None
        self.trix_slope_positive    = None        
        self.rsi_over_upper_limit   = None
        self.rsi_over_lower_limit   = None

    def judge(self,name,index):
        if name == "dc":
            self._judge_dc(index)
    	elif name == "macd":
    		self._judge_macd(index)
    	elif name == "trix":
    		self._judge_trix(index)
        elif name == "rsi":
        	self._judge_rsi(index)


    def _judge_dc(index):
        pass

    def _judge_macd(index):
        pass

    def _judge_trix(index):
        if index.values[-1] > index.values[-2]:
            self.trix_slope_positive = True
            self.trix_slope_negative = not self.trix_slope_positive
        else:
            self.trix_slope_negative = True
            self.trix_slope_positive = not self.trix_slope_negative
        if index.values[-1] > 0:
            self.trix_value_positive = True
            self.trix_value_negative = not self.trix_value_positive
        else:
        	self.trix_value_negative = True
        	self.trix_value_positive = not self.trix_value_negative

        	
    def _judge_rsi(index):
        pass


while kc.activable:
    kc.update()
    

