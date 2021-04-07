#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('train/huobi_usdt_spot_btc_usdt_1min.csv')
from finta import TA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
limit =710
data = TA.CMO(train_df[:limit],100)
y1 = data.dropna()
x1 = np.arange(0, len(y1))
p1 = np.polynomial.Chebyshev.fit(x1, y1, 30)

y_fit1 = p1(x1)
plt.plot(y1)
plt.plot(y_fit1)
plt.show()











from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from finta import TA

def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([('polynomial_features', polynomial_features), ('linear_regression', linear_regression)])
    return pipeline


p_model = polynomial_model(20)
train_df = pd.read_csv('train/huobi_usdt_spot_btc_usdt_1min.csv')

limit = 1000
X = np.linspace(0,1, 2000)[:limit]
data = TA.CMO(train_df,100)[:limit]
data[0] = 0
y = data.values
plt.figure(figsize=(12, 8))


X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
p_model.fit(X, y)
p_model.score(X, y)
plt.plot(X, y)
plt.plot(X, p_model.predict(X), 'r')
plt.show()

