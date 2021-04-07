import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_data(folder):
    dic = dict()
    for file in os.listdir(folder):
        data = pd.read_csv(os.path.join(folder,file))
        filename = file.split(".")[0]
        symbol = filename.split("_")[3]
        dic[symbol]=data['close'][:].values
    return dic


# Creating dataframe with  some different stocks.
data = pd.DataFrame(load_data('data'))
change = data.pct_change()*100
change = change.dropna()

average_change =change.mean()#*60*24

volatility = change.std()#*np.sqrt(60*24)


plt.scatter(volatility,average_change)

names = pd.Series(volatility.index).set_axis(volatility.index)
X = pd.concat([names,volatility,average_change],axis='columns',keys=['Symbol','Volatility','Average change'])

# finding optimum number of cluster using elbow method
from sklearn.cluster import KMeans

w = []
for i in range(1,20):
    km=KMeans(n_clusters=i)
    km.fit(X.iloc[:,[1,2]])
    w.append(km.inertia_)
    
plt.plot(range(1,20),w,marker='o')  

# found optimum number of cluster is 6 approximatly
km = KMeans(n_clusters=6)
y_pred = km.fit_predict(X.iloc[:,[1,2]])
X['Cluster'] = y_pred
X = X.set_index(np.arange(len(volatility)))


# Dataframe showing eack stock and associated cluster
print(X)

# plotting of Clusters
plt.figure(figsize=(15,8))
plt.scatter(X.iloc[y_pred==0,1],X.iloc[y_pred==0,2],c='r',label='Cluster 0',s=100)
plt.scatter(X.iloc[y_pred==1,1],X.iloc[y_pred==1,2],c='g',label='Cluster 1',s=100)
plt.scatter(X.iloc[y_pred==2,1],X.iloc[y_pred==2,2],c='b',label='Cluster 2',s=100)
plt.scatter(X.iloc[y_pred==3,1],X.iloc[y_pred==3,2],c='c',label='Cluster 3',s=100)
plt.scatter(X.iloc[y_pred==4,1],X.iloc[y_pred==4,2],c='m',label='Cluster 4',s=100)
plt.scatter(X.iloc[y_pred==5,1],X.iloc[y_pred==5,2],c='y',label='Cluster 5',s=100)
plt.legend()
plt.xlabel('Volatility')
plt.ylabel('Average change')
for name,vol,ac in zip(X['Symbol'],X['Volatility'],X['Average change']):
    plt.text(vol,ac,name)

plt.show()


