# coding=utf-8

import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def two_mean_list(one, two, type_look='look_min'):
    one_mean = one.mean()
    two_mean = two.mean()
    if type_look == 'look_max':
        one, two = (one, one_mean / two_mean * two) if one_mean > two_mean else (one * two_mean / one_mean, two)
    elif type_look == 'look_min':
        one, two = (one * two_mean / one_mean, two) if one_mean > two_mean else (one, two * one_mean / two_mean)
    return one, two

plt.style.use('fivethirtyeight')

# 利用itertools里的count创建一个迭代器对象,默认从0开始计数, 是一个"无限大"的等差数列
index = count()
x_vals = []
y_vals = []
limit = 1000

def animate(i):
    # i表示的是经历过的"时间", 即每调用一次animate函数, i的值会自动加一
    # 我们从一个动态生成数据的csv文件中获取数据来模拟动态数据
    data1 = pd.read_csv('huobi_usdt_spot_btc_usdt_1min.txt')
    data2 = pd.read_csv('huobi_usdt_swap_eth_usdt_1min.txt')
    # 获得该动态数据的所有列的所有数据(当前生成的), 到下一次运行时该数据如果变动了, 绘出来的图形自然也变动了
    len_data1 = len(data1)
    len_data2 = len(data2)

    x1 = data1.values.transpose()[0][-limit:]
    y1 = data1.values.transpose()[1][-limit:]
    x2 = data2.values.transpose()[0][-limit:]
    y2 = data2.values.transpose()[1][-limit:]
    y1,y2 = two_mean_list(y1,y2)

    # plt对象的cla方法: clear axes: 清除当前轴线(前面说过axes对象表示的是plt整个figure对象下面的一个绘图对象, 一个figure可以有多个axes, 其实就是当前正在绘图的实例).
    # 我们可以不清除当前的axes而沿用前面的axes, 但这样会产生每次绘出来的图形都有很大的变化(原因是重新绘制的时候,颜色,坐标等都重新绘制,可能不在同一个地方了,所以看上去会时刻变化).
    # 因此必须要清除当前axes对象,来重新绘制.
    plt.cla()
    plt.plot(x1, y1, label='spot btc 1')
    plt.plot(x2, y2, label='swap eth 2')
    plt.legend(loc='upper left')
    plt.tight_layout()


# FuncAnimation可以根据给定的interval(时间间隔, ms为单位)一直重复调用某个函数来进行绘制, 从而模拟出实时数据的效果.
ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.show()
