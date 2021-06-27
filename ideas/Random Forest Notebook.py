#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
A random forest classifier aimed at determining whether a stock will be higher or lower after some given amount of days.
Replication of Khaidem, Saha, & Roy Dey (2016)

Documentation on function:
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as make_forest
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import numpy as np
import tqdm

'''
### Outline ###
We have a bunch of columns of different length target values
We drop all target values except the ones we want to analyze (or else when we remove NA we will remove too much data)
We then input the data and features in to the first .fit parameter, and the labels in the second
'''

#Use detrend
selected_data = pd.read_csv('selected_data.csv')
train_labels = ["close_detrend","volume","EWMA", "SO","WR","RSI"]
prediction_data = pd.read_csv('prediction_data.csv')

# In[ ]:


# drop all target columns not to be analyzed
#headers = full_data.columns.values
#headers = headers[13:] # should return just the headers of the target values
#headers = headers[headers!='Target({})'.format(prediction_window)]
#selected_data = full_data.drop(headers, axis=1)


# In[ ]:





# In[ ]:


### Drop useless labels ###
selected_data = selected_data.drop(["index"], axis = 1)
selected_data = selected_data.drop(["date"], axis = 1)
selected_data = selected_data.drop(["open","high","low"], axis = 1)
#selected_data.drop(["symbol","Open","High","Low"], axis = 1, inplace = True)

prediction_data = pd.read_csv('prediction_data.csv')


# In[ ]:
prediction_window = 1
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def split_x_y(df,train_labels,prediction_window):
   
    x = df[train_labels].values
    y = df['Target({})'.format(prediction_window)].values
    
    
    return x,y



n_estimators = 65
oob_score = True 
criterion="gini"
num_features = 6
def train_on_df(x,y,train_frac):
    #Randomly generate array of x lenght 
    msk = np.random.rand(len(x)) < train_frac
    
    train_x = x[msk]
    train_y = y[msk]
    
    test_x = x[~msk]
    test_y = y[~msk]
    
    Random_Forest = make_forest(n_estimators=n_estimators, max_features=num_features, bootstrap=True, oob_score=oob_score, verbose=0,criterion=criterion,n_jobs=-1)
    Random_Forest.fit(train_x, train_y)
    test_accurency = Random_Forest.score(test_x, test_y)
    return Random_Forest, test_accurency


# # Train Modell on each stock and make predictions for 1 and 30 Day
# ## Save them for each Stock into a csv file 

# In[ ]:


stock_forests = {}
import tqdm
num_symboles = len(selected_data['symbol'].unique())-1
for idx,stock in tqdm.tqdm(enumerate(selected_data['symbol'].unique())):
    symbole_df = selected_data[selected_data["symbol"]==stock]
    #x1 == symbole_df[train_labels].values
    #y1 == symbole_df["Target(1)"]
    
    x1,y1 = split_x_y(symbole_df, train_labels,1)
    x30,y30 = split_x_y(symbole_df, train_labels,30)


    forest1, accurency1 = train_on_df(x1,y1,0.8)
    forest30, accurency30 = train_on_df(x30,y30,0.8)


    stock_forests[stock] = {1:{"acc":accurency1,
                                "forest":forest1},
                            30:{"acc":accurency30,
                                "forest":forest30}
                            }

# ## Create File with acc results for all stocks

# In[ ]:


f_all = open("results/_ALL.csv","w")
f_all.write("symbole,accPrediction(1),accPrediction(30)\n")
for symbole, vals in stock_forests.items():
    f_all.write("{},{},{}\n".format(symbole,vals[1]["acc"],vals[30]["acc"]))


# # Train forest over the market

# In[ ]:


x1,y1 = split_x_y(selected_data, train_labels,1)
x30,y30 = split_x_y(selected_data, train_labels,30)

complete_forest1, complete_acc1 = train_on_df(x1,y1,0.8)
complete_forest30, complete_acc30 = train_on_df(x30,y30,0.8)

print("\tacc1: {}%".format(str(round(complete_acc1*100,2))))
print("\tacc30: {}%".format(str(round(complete_acc30*100,2))))





#test predictional data
df_stock = pd.DataFrame()
df_stock["close"] = prediction_data["close"]
df_stock["close_detrend"] = prediction_data["close_detrend"]
# df_stock["Target(1)"] = prediction_data["Target(1)"]
# df_stock["Target(30)"] = prediction_data["Target(30)"]
df_stock["Prediction(1)"] = forest1.predict(prediction_data[train_labels].values)
df_stock["Prediction(30)"] = forest30.predict(prediction_data[train_labels].values)





prediction_data = calculate_targets(prediction_data, 1)
prediction_data = calculate_targets(prediction_data, 3)
prediction_data = calculate_targets(prediction_data, 5)
prediction_data = calculate_targets(prediction_data, 10)
prediction_data = calculate_targets(prediction_data, 14)
prediction_data = calculate_targets(prediction_data, 30)
print('Targets Done - except 60')
prediction_data = prediction_data.dropna(axis=0, how='any')



df_stock["Target(1)"] = prediction_data["Target(1)"]
df_stock["Target(30)"] = prediction_data["Target(30)"]

df_stock = df_stock.dropna(axis=0,how="any")



acc = np.sum(df_stock["Prediction(30)"] == df_stock["Target(30)"] )/len(df_stock)
print("Prediction(30) acc:{}%".format(round(acc,4)*100))
acc = np.sum(df_stock["Prediction(1)"] == df_stock["Target(1)"] )/len(df_stock)
print("Prediction(1) acc:{}%".format(round(acc,4)*100))



