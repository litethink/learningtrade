from product import create_klines
from trade import TradeOperator
kc = create_klines()
to = TradeOperator(capital=10000)
to.init_symbol(kc.symbol,current_price=kc.get_current_price())


# class IndexState(object):
#     """docstring for IndexState"""
#     def __init__(self):
#         self.macd_macd_hold_signal = None
#         self.macd_macd_over_signal = None    
#         self.dc_bottom_slope_positive = None
#         self.dc_bottom_slope_negative = None
#         self.dc_top_slope_positive    = None
#         self.dc_top_slope_negative  = None
#         self.trix_value_negative    = None
#         self.trix_value_positive    = None
#         self.trix_slope_negative    = None
#         self.trix_slope_positive    = None        
#         self.rsi_over_upper_limit   = None
#         self.rsi_over_lower_limit   = None

#     def judge(self,ohlc):
#         macd = ohlc.macd()[0]
#         if macd.MACD[1] > macd.SIGNAL[1] and macd.MACD[0] < macd.SIGNAL[0]:
#             self.macd_macd_hold_signal = True
#         else:
#             self.macd_macd_hold_signal = False
#         if macd.MACD[1] < macd.SIGNAL[1] and macd.MACD[0] > macd.SIGNAL[0]
#             self.macd_macd_over_signal = True
#         else:
#             self.macd_macd_over_signal = False
        



# while kc.activable:
#     kc.update()
    
class IndexState(object):
	"""docstring for ClassName"""
	def __init__(self):
        pass
    def refresh(self):
    	rsi_slow = kc.rsi(14)
    	rsi_fast = kc.rsi(21)
    	macd = kc.macd(fast=26,slow=52,signal=9)
    	if macd.MACD[-1] > macd.SIGNAL[-1]:
    		self.macd_macd_over_signal = True
    		self.macd_macd_hold_signal = not macd_macd_over_signal
        else:
            self.macd_macd_hold_signal = True
            self.macd_macd_over_signal = not macd_macd_hold_signal
        if rsi_slow[-1] > rsi_fast[-1]:
        	self.rsi_slow_over_fast = True
        	self.rsi_slow_hold_fast = not self.rsi_slow_over_fast
        else:
        	self.rsi_slow_hold_fast = True
            self.rsi_slow_over_fast = not self.rsi_slow_hold_fast

        
    def judge(self):
    	if self.trend_positive != True:
    	    if self.macd_macd_hold_signal and self.rsi_slow_over_fast:
    	        self.trend_positive = True
    	else:
    		if self.macd_macd_over_signal:
    			self.trend_positive = False
