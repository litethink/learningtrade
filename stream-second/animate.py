# # coding=utf-8

# import random
# from itertools import count




# # x = np.arange(0, tsla_close.shape[0])



# model = regress_y(y)
# b = model.params[0]
# k = model.params[1]

# y_fit = k * x + b

# plt.plot(x, y)
# plt.plot(x, y_fit, 'r')
# plt.show()
# model.summary()

# #Mean Absolute Error
# MAE = sum(np.abs(y - y_fit)) / len(y)
# #Mean Squared Error
# MSE = sum(np.square(y - y_fit)) / len(y)
# #Root Mean Squard Error
# RMSE = np.sqrt(sum(np.square(y - y_fit)) / len(y))



# """other method"""
# from sklearn import metrics
# MAE = metrics.mean_absolute_error(y,y_fit)
# MSE = metrics.mean_squared_error(y,y_fit)
# RMSE = np.sqrt(metrics.mean_squared_error(y, y_fit))



# import itertools
# _, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
# axs_list = list(itertools.chain.from_iterable(axs))
# poly = np.arange(1, 10, 1)
# for p_cnt, ax in zip(poly, axs_list):
#     p = np.polynomial.Chebyshev.fit(x, y, p_cnt)
#     y_fit = p(x)
#     RMSE = np.sqrt(metrics.mean_squared_error(y, y_fit))

#     ax.set_title('{} poly MSE={}'.format(p_cnt, mse))
#     ax.plot(x, y, '', x, y_fit, 'r.')



# from scipy.interpolate import interp1d, splrep, splev

# _, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

# linear_interp = interp1d(x, y)

# axs[0].set_title('interp1d')

# axs[0].plot(x, y, '', x, linear_interp(x), 'r.')

# splrep_interp = splrep(x, y)

# axs[1].set_title('splrep')

# axs[1].plot(x, y, '', x, splev(x, splrep_interp), 'g.')


def two_mean_list(one, two, type_look='look_min'):
    one_mean = one.mean()
    two_mean = two.mean()
    if type_look == 'look_max':
        one, two = (one, one_mean / two_mean * two) if one_mean > two_mean else (one * two_mean / one_mean, two)
    elif type_look == 'look_min':
        one, two = (one * two_mean / one_mean, two) if one_mean > two_mean else (one, two * one_mean / two_mean)
    return one, two

from sklearn import metrics
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels import regression
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('fivethirtyeight')

# ??????itertools??????count???????????????????????????,?????????0????????????, ?????????"?????????"???????????????

limit = 1000

def regress_y(x,y):
    y = y
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    return model
    
def animate(i):
    data = pd.read_csv('huobi_usdt_spot_btc_usdt_1min.txt')
    data2 = pd.read_csv('huobi_usdt_swap_eth_usdt_1min.txt')
    y = data.values.transpose()[1][-limit:]
    y2 = data2.values.transpose()[1][-limit:]
    y,y2 = two_mean_list(y,y2)
    y = np.array(y,dtype="float64")
    y2 = np.array(y2,dtype="float64")

    x = np.arange(0, len(y))
    x2 = np.arange(0, len(y2))

    model = regress_y(x,y)
    b = model.params[0]
    k = model.params[1]
    # y_fit = k * x + b
    p = np.polynomial.Chebyshev.fit(x, y, 9)
    y_fit = p(x)

    model2 = regress_y(x2,y2)
    b2 = model2.params[0]
    k2 = model2.params[1]
    # y_fit2 = k2 * x2 + b2
    p2 = np.polynomial.Chebyshev.fit(x2, y2, 9)
    y_fit2 = p2(x2)
    #RMSE = np.sqrt(metrics.mean_squared_error(y, y_fit))
    #RMSE2 = np.sqrt(metrics.mean_squared_error(y2, y_fit2))
    # plt?????????cla??????: clear axes: ??????????????????(????????????axes??????????????????plt??????figure?????????????????????????????????, ??????figure???????????????axes, ???????????????????????????????????????).
    # ??????????????????????????????axes??????????????????axes, ???????????????????????????????????????????????????????????????(??????????????????????????????,??????,????????????????????????,??????????????????????????????,??????????????????????????????).
    # ???????????????????????????axes??????,???????????????.
    plt.cla()

    
    plt.plot(x, y,"g", label='btc')
    # plt.plot(x, RMSE,"g", label='btc RMSE')
    plt.plot(x2, y2,"b", label='eth')
    plt.plot(x, y_fit, "r",label='btc fit')   
    # plt.plot(x2, RMSE2, "y",label='eth RMSE')   
    plt.plot(x2, y_fit2, "y",label='eth fit')   

    plt.legend(loc='upper left')
    plt.tight_layout()


# FuncAnimation?????????????????????interval(????????????, ms?????????)?????????????????????????????????????????????, ????????????????????????????????????.
ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.show()
