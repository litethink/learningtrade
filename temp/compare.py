

def plot_two_stock(one, two, axs=None):
	drawer = plt if axs is None else axs
	drawer.plot(one, c='r')
	drawer.plot(two, c='g')
	drawer.grid(True)
	drawer.legend(['one', 'too'], loc='best')

def two_mean_list(one, two, type_look='look_max'):
	one_mean = one.mean()
	two_mean = two.mean()
	if type_look == 'look_max':
		one, two = (one, one_mean / two_mean * two) if one_mean > two_mean else (one * two_mean / one_mean, two)
	elif type_look == 'look_min':
		one, two = (one * two_mean / one_mean, two) if one_mean > two_mean else (one, two * one_mean / two_mean)
	return one, two

def regular_std(group):
	return (group - group.mean()) / group.std()

def regular_mm(group):
	return (group - group.min()) / (group.max() - group.min())


_, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
drawer = axs[0][0]
plot_two_stock(regular_std(df1.close),regular_std(df2.close),
drawer)
drawer.set_title('(group - group.mean()) / group.std()')

drawer = axs[0][1]

plot_two_stock(regular_mm(df1.close),regular_mm(df2.close),drawer)
drawer.set_title(
'(group - group.min()) / (group.max() - group.min())')

drawer = axs[1][0]
one, two = two_mean_list(df1.close, df2.close,
type_look='look_max')
plot_two_stock(one, two, drawer)
drawer.set_title('two_mean_list type_look=look_max')

drawer = axs[1][1]
one, two = two_mean_list(df1.close, df2.close,
type_look='look_min')
plot_two_stock(one, two, drawer)
drawer.set_title('two_mean_list type_look=look_min')

plt.show()




# coding=utf-8

from numpy import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')
limit = 20
x =  [0]
y1 = [0]
y2 = [0]
y3 = [0]
def animate(i):
    x.append(x[-1] + 1 )
    y1.append(random.random())
    y2.append(random.random())
    y3.append(random.random())
    if len(x) > limit:
        x.remove(x[0])   
        y1.remove(y1[0])   
        y2.remove(y2[0])   
        y3.remove(y3[0])   
    plt.cla()
    plt.plot(x, y1, label='Channel 1')
    plt.plot(x, y2, label='Channel 2')
    plt.plot(x, y3, label='Channel 3')
    plt.legend(loc='upper left')
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.show()