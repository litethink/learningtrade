# coding=utf-8

import random
from itertools import count
import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import namedtuple

def visual_section(data, cs_rate ):
    """
        cs_rate: "0.382,0.618..."
        visual_section(data,cs_rate=0.382)
    """
    _cs_max = data.max()
    _cs_min = data.min()
    result = (_cs_max - _cs_min) * cs_rate + _cs_min
    return result


def stat_section(data, cs_rate):
    """
        cs_rate: "0.382,0.618..."
        stat_section(data,cs_rate=0.382)
    """
    cs_rate = cs_rate * 100
    result = scoreatpercentile(data, cs_rate)
    return result



def two_mean_list(one, two, type_look='look_max'):
    one_mean = one.mean()
    two_mean = two.mean()
    if type_look == 'look_max':
        one, two = (one, one_mean / two_mean * two) if one_mean > two_mean else (one * two_mean / one_mean, two)
    elif type_look == 'look_min':
        one, two = (one * two_mean / one_mean, two) if one_mean > two_mean else (one, two * one_mean / two_mean)
    return one, two


def derepeat(data):
    y =[data[0]]
    for i in range(len(data)-1):
        if data[i][0] ==  data[i+1][0]:
            y.append(data[i+1])
        else:
            if data[i][0] != y[-1][0]:
                y.append(data[i])

    if data[-1][0] != y[-1][0]:
        y.append(data[-1])
    data = np.array(y).transpose()[1]
    return data
            # if data[-1][0] != y[-1][0]:
            #     y.append(data[-1])

# plt.style.use('fivethirtyeight')
limit = 1000
polynome=6
def animate(i):
    data1 = pd.read_csv('huobi_usdt_spot_btc_usdt_1min.txt')
    data2 = pd.read_csv('huobi_usdt_swap_eth_usdt_1min.txt')
    if len(data1) > limit and len(data1) >limit:
        y1 = data1.values.transpose()[1][-limit:]
        y2 = data2.values.transpose()[1][-limit:]


        y1 = np.array(y1,dtype="float64")[-limit:]
        y2 = np.array(y2,dtype="float64")[-limit:]
        y1,y2 = two_mean_list(y1,y2)
        y3 = (y1 + y2)/2
        x1 = np.arange(0, len(y1))
        x2 = np.arange(0, len(y2))
        x3 = np.arange(0, len(y3))

        p1 = np.polynomial.Chebyshev.fit(x1, y1, polynome)
        y_fit1 = p1(x1)

        p2 = np.polynomial.Chebyshev.fit(x2, y2, polynome)
        y_fit2 = p2(x2)

        p3 = np.polynomial.Chebyshev.fit(x3, y3, polynome)
        y_fit3 = p3(x3)

        vs382 = visual_section(y1,cs_rate=0.382)
        vs618 = visual_section(y1,cs_rate=0.618)
        ss382 = stat_section(y1,cs_rate=0.382)
        ss618 = stat_section(y1,cs_rate=0.618)

        def plot_golden():
            above618 = np.maximum(vs618, ss618)
            below618 = np.minimum(vs618, ss618)
            above382 = np.maximum(vs382, ss382)
            below382 = np.minimum(vs382, ss382)
            
            plt.axhline(vs382, c='r')
            plt.axhline(ss382, c='m')
            plt.axhline(vs382, c='g')
            plt.axhline(ss618, c='k')
            plt.fill_between(x1, above618, below618,
                             alpha=0.5, color="r")
            plt.fill_between(x1, above382, below382,
                             alpha=0.5, color="g")
            return namedtuple('golden', ['above618', 'below618', 'above382',
                                         'below382'])(
                above618, below618, above382, below382)

        plt.cla()
        golden = plot_golden()
        plt.plot(x1,y1, "r", label='btc spot')
        plt.plot(x2,y2, "g", label='eth swap')
        plt.plot(x3,y3, "y", label="mean two")
        plt.plot(x1,y_fit1, "k", label="fit btc")
        plt.plot(x2,y_fit2, "c", label="fit eth")
        plt.plot(x3,y_fit3, "m", label="fit mean")
        plt.legend(loc='upper left')

ani = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.show()
