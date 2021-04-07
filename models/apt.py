#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats


# ### A. Calculate historical factor monthly returns for the following factors based on APT.

# In[2]:


# Load data of Monthly Returns for S&P 500 Index and US Dollar Index.
df_SPD = pd.read_excel('APT&BLM.xlsx',sheet_name = 'SPX_DXY_MonthlyReturns',parse_dates=[0],index_col=[0])
df_SPD


# In[3]:

#daily close
df_500 = pd.read_excel('APT&BLM.xlsx',sheet_name = 'SP500Stocks',index_col=[0],skiprows=1)
df_500


# In[4]:


Beta_mat1 = np.zeros(shape=(505,3))


# In[5]:


# Run a regression using equity market factor and US Dollar factor. Get the Beta matrix.
for i in range(len(df_500)):
    df_stock = df_500.iloc[i]
    df_stockret = df_stock[3:].astype('float')
    df_stockret = df_stockret.to_frame()
    df_step1 = pd.concat([df_stockret,df_SPD],axis=1)
    df_step1.dropna(inplace=True)
    df_step1.columns = ['Stock','SP','USD']
    list1 = df_step1.columns.tolist()
    formula1 = str(list1[0])+' ~ '+str(list1[1])
    del list1[0:2]
    for n in list1:
        formula1 += '+'
        formula1 += str(n)

    result1 = smf.ols(formula1,data=df_step1).fit()
    Beta_mat1[i]=result1.params.values


# In[6]:
#Beta matrix
#columns : close,S&P,US Dollar
#rows: stock or symbol 
#[505 rows x 3 columns]
pd.DataFrame(Beta_mat1)


# In[7]:


# stock Sector factors. 
sector = df_500.iloc[:,1]
sector


# In[8]:


# Leaving out the UTIL.
sector1 = pd.get_dummies(df_500, columns = ['SECTOR'])
sector1.drop('SECTOR_UTIL',axis=1,inplace=True)
sector1.iloc[:,-11:]


# In[9]:


# Get sector exposures.
Beta_mat2 = Beta_mat1
sector_expo = sector1.iloc[:,-10:]
sector_expo


# In[10]:


# Put the sector exposures with Beta matrix 1 to get Beta matrix 2.
Beta_mat2 = pd.DataFrame(Beta_mat1,columns=['coef','beta_SP','beta_USD'],index=sector_expo.index)
Beta_mat2 = pd.concat([Beta_mat2,sector_expo],axis=1)
#leave out coef column
Beta_mat2 = Beta_mat2.iloc[:,1:]
Beta_mat2


# In[11]:


factor_ret = np.zeros(shape=(13,210))


# In[12]:


# Size factor
# column: SECTOR ,Market Cap ($ Mil)
size_factor = df_500.iloc[:,[1,2]]
#log 'Market Cap ($ Mil)'
size_factor['Market Cap ($ Mil)'] = np.log(size_factor['Market Cap ($ Mil)'])
size_factor


# In[13]:

#the mean the sactor of Market Cap ($ Mil) 
size_sec = size_factor.groupby('SECTOR').mean()
size_sec.columns = ['MarketCapMean']
size_sec

# Out[0]:
#size_sec
#        MarketCapMean
#SECTOR               
#DSCR         9.668220
#ENER         9.922564
#FINA        10.069093
#HLTH        10.382486
#INDU         9.887828
#INFT        10.225671
#$MATS         9.724490
#REAL         9.682977
#STPL        10.415517
#TCOM        10.488791
#UTIL         9.946875


# In[14]:


size_sec['MarketCapStd'] = size_factor.groupby('SECTOR').std()
size_sec


# In[15]:


df_500_1 = df_500.copy()
for i in size_sec.index:
    df_500_1.loc[(df_500_1['SECTOR']==i),'MarketCapMean']=size_sec.loc[i,'MarketCapMean']
    df_500_1.loc[(df_500_1['SECTOR']==i),'MarketCapStd']=size_sec.loc[i,'MarketCapStd']

df_500_1


# In[16]:


# Calculate z-score of log(mkt_cap).
df_500_1['zscore']=(np.log(df_500_1['Market Cap ($ Mil)'])-df_500_1['MarketCapMean'])/df_500_1['MarketCapStd']
df_500_1


# In[17]:


# Put size factor with Beta Matrix to get the final Beta Matrix.
Beta_mat2 = pd.concat([Beta_mat2,df_500_1['zscore']],axis=1)
Beta_mat2


# In[18]:


# Run cross-sectional regression.
for i in range(210):
    reg = pd.concat([df_500.iloc[:,i+3],Beta_mat2],axis=1)
    reg.dropna(inplace=True)
    X = reg.iloc[:,-13:]
    X = sm.add_constant(X)
    Y = reg.iloc[:,0]

    result2 = sm.OLS(Y,X).fit()
    result2.summary()
    factor_ret[:,i] = result2.params.values[1:]
    
factor_ret


# In[19]:


pd.DataFrame(factor_ret)


# ### B. Check for each factor their historical returns are significant or not (based on T-stat).

# In[20]:


# B
for i in range(13):
    fac = factor_ret[i,:]
    print(stats.ttest_1samp(fac, 0))
    


# #### The results show that pvalues are very large (>0.05), so the null hypothesis cannot be rejected. So the historical returns are not significant.

# ### C. Using the last month in the back-test, i.e., 12/31/2018:
# ### 1. all factor portfolios are long-short neutral portfolio, i.e., the total weights sum to 0.

# In[21]:


# Add one column for coefficients of alpha.
Beta_mat3 = sm.add_constant(Beta_mat2)
Beta_mat3


# In[22]:


mat_Beta = np.mat(Beta_mat3)
factor_port = ((mat_Beta.T)*mat_Beta).I*mat_Beta.T
pd.DataFrame(factor_port)


# In[23]:


factor_port.shape


# In[24]:


sumweight = factor_port.sum(axis=1)
sumweight[1:]


# In[25]:


# All the sums of weights for each factor are close to 0. So all factor portfolios are long-short neutral portfolio.


# ### 2. For any factor portfolio, it has unit exposure to its own factor, but zero exposure to all other factors in the model.

# In[26]:


for i in range(1,14):
    expoitself = mat_Beta[:,i].T*factor_port[i,:].T
    print(expoitself)


# In[27]:


# For any factor portfolio, it has unit exposure to its own factor (equal to 1).


# In[28]:


# Say for the first factor.
for i in range(2,14):
    otherexpo = mat_Beta[:,i].T*factor_port[1,:].T
    print(otherexpo)


# In[29]:


# Calculate the exposure matrix.
exposure_matrix = np.zeros(shape=(13,13))
for factor_num in range(1,14):
    for i in range(1,14):
        exposure_matrix[factor_num-1, i-1] = mat_Beta[:,i].T*factor_port[factor_num,:].T

exposure_matrix = pd.DataFrame(exposure_matrix)
exposure_matrix


# In[30]:


# For any factor portfolio, it has zero exposure to all other factors in the model. All close to 0.

