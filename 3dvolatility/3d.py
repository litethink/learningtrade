from datetime import datetime
import time
import pandas as pd
import numpy as np
day = datetime.utcfromtimestamp(time.time()).day

lengh = 500


def regular_std(group):
# z-score
    return (group - group.mean()) / group.std()




def gen_data():
    data = pd.read_csv('huobi_usdt_spot_btc_usdt_1min_{}.txt'.format(day))
    changes = np.array([[0.0,0.0,0.0]])

    for i in range(lengh):
        single = data.iloc[-(2*lengh - i): -(lengh - i),1]

        change = single.pct_change()*1000
        change = change.dropna()
        average_change =change.mean()
        volatility = change.std()#
        changes = np.insert(changes,len(changes),(i,float(average_change),volatility),axis=0)

    last = data.iloc[-lengh:,1]
    change = last.pct_change()*1000
    change = change.dropna()
    volatility = change.std()#
    average_change =change.mean()
    changes = np.insert(changes,len(changes),(i,average_change,volatility),axis=0)
    return changes
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from itertools import count
from mpl_toolkits.mplot3d import Axes3D




ax = plt.axes(projection = '3d')
ax.set_xlabel('time')
ax.set_title('volatility')
ax.set_ylabel('price')
#ax.set_xlim([0,70])
#ax.set_ylim([-60,-70])

plt.grid(True)
plt.ion()  # interactive mode on!!!! 很重要，有了他就不需要plt.show()了

for t in count():
    if t == 2500:
        break
    arr_changes = gen_data()
    plt.cla() # 此命令是每次清空画布，所以就不会有前序的效果
    ax.plot3D(arr_changes[:, 0],arr_changes[:, 2], arr_changes[:, 1], 'y')
    if t%360 == 0: 
        last_change = arr_changes[-1]
        print ('time:{},change:{},volatility:{}'.format(last_change[0],last_change[1],last_change[2]))
    plt.pause(0.001)
