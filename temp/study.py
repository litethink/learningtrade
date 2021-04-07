from collections import namedtuple
import numpy as np
from scipy.stats import scoreatpercentile
from matplotlib import pyplot as plt

#cd ~/backup/test_index

from product import create_klines
from trade import TradeOperator
kc = create_klines()
to = TradeOperator(capital=10000)
to.init_symbol(kc.symbol,current_price=kc.get_current_price())
df = kc._ohlc


"""Fibonacci"""
def visual_section(df, cs_rate, column="close"):
    """
        cs_rate: "0.382,0.618..."
        visual_section(df,column="close",cs_rate=0.382)
        visual_section(df,column="close",cs_rate=0.618)
    """
    _column = df[[column]]
    _cs_max = _column.max()[0]
    _cs_min = _column.min()[0]
    result = (_cs_max - _cs_min) * cs_rate + _cs_min
    return result


def stat_section(df, cs_rate, column="close"):
    """
        cs_rate: "0.382,0.618..."
        stat_section(df,column="close",cs_rate=0.382)
        stat_section(df,column="close",cs_rate=0.618)
    """
    cs_rate = cs_rate * 100
    _column = df[[column]] 
    result = scoreatpercentile(_column, cs_rate)
    return result


if __name__ == '__main__':


    vs382 = visual_section(df,column="close",cs_rate=0.382)
    vs618 = visual_section(df,column="close",cs_rate=0.618)
    ss382 = stat_section(df,column="close",cs_rate=0.382)
    ss618 = stat_section(df,column="close",cs_rate=0.618)


    def plot_golden():
        above618 = np.maximum(vs618, ss618)
        below618 = np.minimum(vs618, ss618)
        above382 = np.maximum(vs382, ss382)
        below382 = np.minimum(vs382, ss382)
        
        plt.plot(df.close)
        plt.axhline(vs382, c='r')
        plt.axhline(ss382, c='m')
        plt.axhline(vs382, c='g')
        plt.axhline(ss618, c='k')
        plt.fill_between(df.index, above618, below618,
                         alpha=0.5, color="r")
        plt.fill_between(df.index, above382, below382,
                         alpha=0.5, color="g")
        return namedtuple('golden', ['above618', 'below618', 'above382',
                                     'below382'])(
            above618, below618, above382, below382)

    golden = plot_golden()
    plt.legend(['close', 'sp382', 'sp382_stats', 'sp618', 'sp618_stats'],
               loc='best')

    plt.show()



"""Fitting Degree and Regression"""

import statsmodels.api as sm
from statsmodels import regression
from matplotlib import pyplot as plt
tsla_close = df.close
x = np.arange(0, tsla_close.shape[0])
y = tsla_close.values

def regress_y(y):
	y = y
	x = np.arange(0, len(y))
	x = sm.add_constant(x)
	model = regression.linear_model.OLS(y, x).fit()
	return model
model = regress_y(y)
b = model.params[0]
k = model.params[1]

y_fit = k * x + b

plt.plot(x, y)
plt.plot(x, y_fit, 'r')
plt.show()
model.summary()

#Mean Absolute Error
MAE = sum(np.abs(y - y_fit)) / len(y)
#Mean Squared Error
MSE = sum(np.square(y - y_fit)) / len(y)
#Root Mean Squard Error
RMSE = np.sqrt(sum(np.square(y - y_fit)) / len(y))



"""other method"""
from sklearn import metrics
MAE = metrics.mean_absolute_error(y,y_fit)
MSE = metrics.mean_squared_error(y,y_fit)
RMSE = np.sqrt(metrics.mean_squared_error(y, y_fit))


import numpy as np
import itertools
_, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
axs_list = list(itertools.chain.from_iterable(axs))
poly = np.arange(1, 10, 1)
for p_cnt, ax in zip(poly, axs_list):
	p = np.polynomial.Chebyshev.fit(x, y, p_cnt)
	y_fit = p(x)
	RMSE = np.sqrt(metrics.mean_squared_error(y, y_fit))

	ax.set_title('{} poly MSE={}'.format(p_cnt, RMSE))
	ax.plot(x, y, '', x, y_fit, 'r.')



from scipy.interpolate import interp1d, splrep, splev

_, axs = plt.subplots(nrows=1, ncols=1, figsize=(14, 3))

linear_interp = interp1d(x, y)

axs[0].set_title('interp1d')

axs[0].plot(x, y, '', x, linear_interp(x), 'r.')

splrep_interp = splrep(x, y)

axs[1].set_title('splrep')

axs[1].plot(x, y, '', x, splev(x, splrep_interp), 'g.')


plt.show()


"""MonteCarlo method"""

from abc import ABCMeta, abstractmethod
import six
import numpy as np
K_INIT_LIVING_DAYS = 75*365
class Person(object):
	def __init__(self):
		self.living = K_INIT_LIVING_DAYS
		self.happiness = 0
		self.wealth = 0
		self.fame = 0
		self.living_day = 0
	def live_one_day(self, seek):
		consume_living, happiness, wealth, fame = seek.do_seek_day()
		self.living -= consume_living
		self.happiness += happiness
		self.wealth += wealth
		self.fame += fame
		self.living_day += 1

class BaseSeekDay(six.with_metaclass(ABCMeta, object)):
	def __init__(self):
		self.living_consume = 0
		self.happiness_base = 0
		self.wealth_base = 0
		self.fame_base = 0
		self.living_factor = [0]
		self.happiness_factor = [0]
		self.wealth_factor = [0]
		self.fame_factor = [0]
		self.do_seek_day_cnt = 0
		self._init_self()
	@abstractmethod
	def _init_self(self, *args, **kwargs):
		pass
	@abstractmethod
	def _gen_living_days(self, *args, **kwargs):
		pass
	def do_seek_day(self):
		if self.do_seek_day_cnt >= len(self.living_factor):
			consume_living = self.living_factor[-1] * self.living_consume
		else:
			consume_living = self.living_factor[self.do_seek_day_cnt] * self.living_consume
		if self.do_seek_day_cnt >= len(self.happiness_factor):
			happiness = self.happiness_factor[-1] * self.happiness_base
		else:
			happiness = self.happiness_factor[self.do_seek_day_cnt] * self.happiness_base
		if self.do_seek_day_cnt >= len(self.wealth_factor):
			wealth = self.wealth_factor[-1] * self.wealth_base
		else:
			wealth = self.wealth_factor[ self.do_seek_day_cnt] * self.wealth_base 
		if self.do_seek_day_cnt >= len(self.fame_factor):
			fame = self.fame_factor[-1] * self.fame_base
		else:
			fame = self.fame_factor[self.do_seek_day_cnt] * self.fame_base
		self.do_seek_day_cnt += 1
		return consume_living, happiness, wealth, fame


def regular_mm(group):
# 最 -最大规范化
	return (group - group.min()) / (group.max() - group.min())


class HealthSeekDay(BaseSeekDay):
	def _init_self(self):
		self.living_consume = 1
		self.happiness_base = 1
		self._gen_living_days()

	def _gen_living_days(self):
		days = np.arange(1, 12000)
		living_days = np.sqrt(days)
		self.living_factor = regular_mm(living_days) * 2 - 1
		self.happiness_factor = regular_mm(days)[::-1]




me = Person()
seek_health = HealthSeekDay()
while me.living > 0:
	me.live_one_day(seek_health)
print('living:{} years, happiness:{}, wealth:{}, fame:{}'.format(round(me.living_day / 365, 2), round(me.happiness,2),me.wealth, me.fame))



class StockSeekDay(BaseSeekDay):

	def _init_self(self, show=False):
		self.living_consume = 2
		self.happiness_base = 0.5
		self.wealth_base = 10
		self._gen_living_days()
	def _gen_living_days(self):
		days = np.arange(1, 10000)
		living_days = np.sqrt(days)
		self.living_factor = regular_mm(living_days)
		happiness_days = np.power(days, 4)
		self.happiness_factor = regular_mm(happiness_days)[::-1]
		self.wealth_factor = self.living_factor


me = Person()

seek_stock = StockSeekDay()
while me.living > 0:
	me.live_one_day(seek_stock)

print('living:{} years, happiness:{}, wealth:{}, fame:{}'
	.format(round(me.living_day / 365, 2), round(me.happiness,2),me.wealth, me.fame))


class FameSeekDay(BaseSeekDay):
	def _init_self(self):
		self.living_consume = 3
		self.happiness_base = 0.6
		self.fame_base = 10
		self._gen_living_days()
	def _gen_living_days(self):
		days = np.arange(1, 12000)
		living_days = np.sqrt(days)
		self.living_factor = regular_mm(living_days)
		happiness_days = np.power(days, 2)
		self.happiness_factor = regular_mm(happiness_days)[::-1]
		self.fame_factor = self.living_factor

me = Person()
seek_fame = FameSeekDay()
while me.living > 0:
	me.live_one_day(seek_fame)
print('living:{} years, happiness:{}, wealth:{}, fame:{}'
	.format(round(me.living_day / 365, 2), round(me.happiness,2), round(me.wealth, 2), round(me.fame, 2)))




def my_life(weights):
	seek_health = HealthSeekDay()
	seek_stock = StockSeekDay()
	seek_fame = FameSeekDay()
	seek_list = [seek_health, seek_stock, seek_fame]
	me = Person()
	seek_choice = np.random.choice([0, 1, 2], 80000,p=weights)
	while me.living > 0:
		seek_ind = seek_choice[me.living_day]
		seek = seek_list[seek_ind]
		me.live_one_day(seek)
	return round(me.living_day / 365, 2),round(me.happiness, 2),round(me.wealth, 2), round(me.fame, 2)


weights=[0.3,0.5,0.2]
my_life(weights)

weights = np.random.dirichlet(np.ones(3),size=1)[0]
my_life(weights)

result = []
for _ in range(2000):
	weights = np.random.dirichlet(np.ones(3),size=1)[0]
	result.append((weights, my_life(weights)))

sorted_scores = sorted(result, key=lambda x: x[1][1],reverse=True)
sorted_scores[0]
sorted_scores = sorted(result, key=lambda x: x[1][2],reverse=True)
sorted_scores[0]
sorted_scores = sorted(result, key=lambda x: x[1][3],reverse=True)
sorted_scores[0]
sorted_scores = sorted(result, key=lambda x: x[1][0],reverse=True)
sorted_scores[0]




import scipy.optimize as sco
from scipy.interpolate import interp1d
tsla_close = df.close
# x = (0, 1, 2,...,len(tsla_close))
x = np.arange(0, tsla_close.shape[0])
y = tsla_close.values
linear_interp = interp1d(x, y)
plt.plot(linear_interp(x))
global_min_pos = sco.fminbound(linear_interp, 1, 504)
plt.plot(global_min_pos, linear_interp(global_min_pos),'r<')

last_postion = None
for find_min_pos in np.arange(50, len(x), 50):
	local_min_pos = sco.fmin_bfgs(linear_interp,find_min_pos, disp=0)
	draw_postion = (local_min_pos,linear_interp(local_min_pos))
	if last_postion is not None:
		plt.plot([last_postion[0][0], draw_postion[0][0]],[last_postion[1][0], draw_postion[1][0]],'o-')
	last_postion = draw_postion


def minimize_happiness_global(weights):
	if np.sum(weights) != 1:
		return 0
	return - my_life(weights)[1]


opt_global = sco.brute(minimize_happiness_global,((0, 1.1, 0.1), (0, 1.1, 0.1), (0,1.1, 0.1)))

my_life(opt_global)


def minimize_happiness_local(weights):
	print(weights)
	return -my_life(weights)[1]

method='SLSQP'
bounds = tuple((0, 1) for x in range(3))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) -
1})
guess = [0.5, 0.2, 0.3]
opt_local = sco.minimize(minimize_happiness_local, guess,
method=method, bounds=bounds,
constraints=constraints)


