import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint 
from scipy.stats import pearsonr
from utils import *
data = pd.read_excel("futures.xlsx")
_, pv_coint, _ = coint(data['CU.SHF'], data['SF.CZC'])
corr, pv_corr = pearsonr(data['CU.SHF'], data['SF.CZC'])
print("Cointegration pvalue : %0.4f"%pv_coint)
print("Correlation coefficient is %0.4f and pvalue is %0.4f"%(corr, pv_corr))

plt.plot(normal_data(data["CU.SHF"]))
plt.plot(normal_data(data["SF.CZC"]))

S1 = data["CU.SHF"]
S2 = data["SF.CZC"]

NS1 = normal_data(data["CU.SHF"])
NS2 = normal_data(data["SF.CZC"])
ratios1 = S1 / S2 

ratios2 = NS1/NS2

ratios1.hist(bins = 200)
ratios2.hist(bins = 200)

#静态zscore
zscore = (ratios1 - ratios1.mean()) / ratios1.std()


#动态调整
ratios_mavg1 = ratios1.rolling(window=5, center=False).mean()
ratios_mavg2 = ratios1.rolling(window=60, center=False).mean()
std = ratios1.rolling(window=60, center=False).std()
zscore_mv = (ratios_mavg1 - ratios_mavg2) / std

#动态调整后的zscore和市价的关系展示
plt.plot(NS1,"r")
plt.plot(NS2,"b")
plt.plot(zscore_mv)

#zscore和市价的关系展示
plt.plot(NS1,"r")
plt.plot(NS2,"b")
plt.plot(zscore)

